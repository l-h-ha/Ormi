#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <latch>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/float16.h>
#include <ormi/core/memory/ops.h>
#include <ormi/core/op.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/tensor.h>

#include "backends/cuda/fusion/tile.h"
#include "backends/cuda/state.h"
#include "core/runtime/device.h"
#include "core/runtime/fusion/expr.h"

namespace {
    using namespace ormi::core;
    using TensorPtr = std::shared_ptr<Tensor>;
    using Graph = std::unique_ptr<CUgraph_st, decltype(&cuGraphDestroy)>;

    void require(CUresult status) {
        if (status != CUDA_SUCCESS) {
            const char* message = nullptr;
            (void)cuGetErrorString(status, &message);
            throw std::runtime_error(message ? message : "CUDA graph query failed.");
        }
    }

    Graph capture(const Setup& setup, Tensor& output) {
        const auto stream = reinterpret_cast<CUstream>(
            ormi::backends::cuda::stream(output.device())
        );
        require(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
        CUgraph graph = nullptr;
        try {
            setup.run(output);
        } catch (...) {
            (void)cuStreamEndCapture(stream, &graph);
            if (graph) {
                (void)cuGraphDestroy(graph);
            }
            throw;
        }
        const auto status = cuStreamEndCapture(stream, &graph);
        Graph result(graph, cuGraphDestroy);
        require(status);
        return result;
    }

    std::vector<CUDA_KERNEL_NODE_PARAMS> kernels(const Graph& graph) {
        std::size_t count = 0;
        require(cuGraphGetNodes(graph.get(), nullptr, &count));
        std::vector<CUgraphNode> nodes(count);
        require(cuGraphGetNodes(graph.get(), nodes.data(), &count));
        std::vector<CUDA_KERNEL_NODE_PARAMS> result;
        for (const auto node : nodes) {
            CUgraphNodeType type;
            require(cuGraphNodeGetType(node, &type));
            if (type != CU_GRAPH_NODE_TYPE_KERNEL) {
                throw std::runtime_error("Fusion capture contains a non-kernel node.");
            }
            CUDA_KERNEL_NODE_PARAMS parameters{};
            require(cuGraphKernelNodeGetParams(node, &parameters));
            result.push_back(parameters);
        }
        return result;
    }

    unsigned int rounded_threads(std::size_t values) {
        unsigned int threads = 1;
        while (threads < 256 && threads < values) {
            threads *= 2;
        }
        return threads;
    }

    unsigned int available_threads(CUfunction function) {
        int minimum_blocks = 0;
        int threads = 0;
        require(cuOccupancyMaxPotentialBlockSize(
            &minimum_blocks, &threads, function, nullptr, 0, 0
        ));
        unsigned int result = 1;
        while (result <= static_cast<unsigned int>(threads) / 2) {
            result *= 2;
        }
        return result;
    }

    void expect_dimensions(const CUDA_KERNEL_NODE_PARAMS& parameters) {
        EXPECT_GT(parameters.gridDimX, 0u);
        EXPECT_EQ(parameters.gridDimY, 1u);
        EXPECT_EQ(parameters.gridDimZ, 1u);
        EXPECT_GT(parameters.blockDimX, 0u);
        EXPECT_EQ(parameters.blockDimY, 1u);
        EXPECT_EQ(parameters.blockDimZ, 1u);
        EXPECT_NE(parameters.kernelParams, nullptr);
    }

    void record_launch(
        const std::string& name,
        const CUDA_KERNEL_NODE_PARAMS& parameters,
        std::size_t count_index
    ) {
        const auto count = *static_cast<const unsigned long long*>(
            parameters.kernelParams[count_index]
        );
        ::testing::Test::RecordProperty(name,
            std::to_string(parameters.gridDimX) + "/" +
            std::to_string(parameters.blockDimX) + "/" +
            std::to_string(parameters.sharedMemBytes) + "/" +
            std::to_string(count)
        );
    }

    class CUDAFusionExecutionTest : public ::testing::TestWithParam<DType> {
    protected:
        void SetUp() override {
            int devices = 0;
            const auto status = cudaGetDeviceCount(&devices);
            if (status != cudaSuccess) {
                GTEST_SKIP() << cudaGetErrorString(status);
            }
            if (devices == 0) {
                GTEST_SKIP() << "No CUDA device is available.";
            }
        }

        TensorPtr tensor(Shape shape, double value = 0.5, Device device = Device::from("cuda")) const {
            Scalar scalar;
            switch (GetParam()) {
                case DType::float16: scalar = Float16(static_cast<float>(value)); break;
                case DType::float32: scalar = static_cast<float>(value); break;
                case DType::float64: scalar = value; break;
                default: throw std::runtime_error("Expected a floating-point dtype.");
            }
            auto result = ops::full(std::move(shape), scalar, device);
            result->realize();
            return result;
        }

        static TensorPtr mapped(const TensorPtr& input) {
            return ops::add(ops::mul(input, input), input);
        }

        static ormi::core::Setup prepare(const TensorPtr& graph, Tensor& output) {
            const auto expression = runtime::fusion::build(*graph);
            if (!expression) {
                throw std::runtime_error("Cannot build fusion expression.");
            }
            return runtime::backend(output.device()).fusion.prepare(
                output, {}, *expression
            );
        }

        static double read(Tensor& tensor) {
            switch (tensor.dtype()) {
                case DType::float16: {
                    Float16 value;
                    memory::copy(&value, Device::cpu(), tensor.data(), tensor.device(), sizeof(value));
                    return static_cast<float>(value);
                }
                case DType::float32: {
                    float value;
                    memory::copy(&value, Device::cpu(), tensor.data(), tensor.device(), sizeof(value));
                    return value;
                }
                case DType::float64: {
                    double value;
                    memory::copy(&value, Device::cpu(), tensor.data(), tensor.device(), sizeof(value));
                    return value;
                }
                default: throw std::runtime_error("Expected a floating-point dtype.");
            }
        }

        std::size_t accumulator_bytes() const {
            return GetParam() == DType::float64 ? sizeof(double) : sizeof(float);
        }
    };

    TEST_P(CUDAFusionExecutionTest, pointwiseLaunchBindsInputsAndElementCount) {
        for (const auto elements : {1, 257, 1048576}) {
            const auto lhs = tensor({elements});
            const auto rhs = tensor({elements}, 0.25);
            const auto output = tensor({elements}, 0.0);
            const auto graph = ops::add(ops::mul(lhs, lhs), rhs);
            auto setup = prepare(graph, *output);
            EXPECT_EQ(setup.buffers(), 0);
            EXPECT_EQ(setup.bytes(), 0);
            const auto captured = capture(setup, *output);
            const auto nodes = kernels(captured);
            ASSERT_EQ(nodes.size(), 1);
            const auto& main = nodes[0];
            expect_dimensions(main);
            EXPECT_EQ(main.blockDimX, available_threads(main.func));
            EXPECT_EQ(main.sharedMemBytes, 0u);
            int active_blocks = 0;
            require(cuOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks, main.func, main.blockDimX, 0
            ));
            int multiprocessors = 0;
            ASSERT_EQ(cudaDeviceGetAttribute(
                &multiprocessors, cudaDevAttrMultiProcessorCount, output->device().index()
            ), cudaSuccess);
            const auto required = (elements + main.blockDimX - 1) / main.blockDimX;
            EXPECT_EQ(main.gridDimX, std::min(required,
                static_cast<unsigned int>(std::max(1, multiprocessors * active_blocks))
            ));
            EXPECT_EQ(*static_cast<void**>(main.kernelParams[0]), output->data());
            EXPECT_EQ(*static_cast<const void**>(main.kernelParams[1]), lhs->data());
            EXPECT_EQ(*static_cast<const void**>(main.kernelParams[2]), rhs->data());
            EXPECT_EQ(*static_cast<unsigned long long*>(main.kernelParams[3]), elements);
            record_launch("main_" + std::to_string(elements), main, 3);
            setup.run(*output);
            EXPECT_EQ(read(*output), 0.5);
        }
    }

    TEST_P(CUDAFusionExecutionTest, reusesCompiledFunctionWithDifferentTensorAddresses) {
        const auto first_input = tensor({263});
        const auto second_input = tensor({263}, 0.25);
        const auto first_output = tensor({263}, 0.0);
        const auto second_output = tensor({263}, 0.0);
        auto first = prepare(mapped(first_input), *first_output);
        auto second = prepare(mapped(second_input), *second_output);
        const auto first_graph = capture(first, *first_output);
        const auto second_graph = capture(second, *second_output);
        const auto first_nodes = kernels(first_graph);
        const auto second_nodes = kernels(second_graph);
        ASSERT_EQ(first_nodes.size(), 1u);
        ASSERT_EQ(second_nodes.size(), 1u);
        EXPECT_EQ(first_nodes[0].func, second_nodes[0].func);
        EXPECT_EQ(*static_cast<const void**>(second_nodes[0].kernelParams[1]), second_input->data());

        first.run(*first_output);
        second.run(*second_output);
        EXPECT_DOUBLE_EQ(read(*first_output), 0.75);
        EXPECT_DOUBLE_EQ(read(*second_output), 0.3125);
    }

    TEST_P(CUDAFusionExecutionTest, keepsDifferentFusionExpressionsSeparate) {
        const auto input = tensor({267});
        const auto output = tensor({267}, 0.0);
        const auto added = mapped(input);
        const auto subtracted = ops::sub(ops::mul(input, input), input);
        auto first = prepare(added, *output);
        auto second = prepare(subtracted, *output);
        auto repeated = prepare(added, *output);
        const auto first_graph = capture(first, *output);
        const auto second_graph = capture(second, *output);
        const auto repeated_graph = capture(repeated, *output);
        const auto first_nodes = kernels(first_graph);
        const auto second_nodes = kernels(second_graph);
        const auto repeated_nodes = kernels(repeated_graph);
        ASSERT_EQ(first_nodes.size(), 1u);
        ASSERT_EQ(second_nodes.size(), 1u);
        ASSERT_EQ(repeated_nodes.size(), 1u);
        EXPECT_NE(first_nodes[0].func, second_nodes[0].func);
        EXPECT_EQ(first_nodes[0].func, repeated_nodes[0].func);

        second.run(*output);
        EXPECT_DOUBLE_EQ(read(*output), -0.25);
        repeated.run(*output);
        EXPECT_DOUBLE_EQ(read(*output), 0.75);
    }

    TEST_P(CUDAFusionExecutionTest, publishesOneFunctionAcrossConcurrentPreparation) {
        constexpr std::size_t workers = 4;
        std::vector<TensorPtr> inputs;
        std::vector<TensorPtr> outputs;
        std::vector<TensorPtr> graphs;
        for (std::size_t worker = 0; worker < workers; ++worker) {
            inputs.push_back(tensor({269}, 0.5 + static_cast<double>(worker) * 0.125));
            outputs.push_back(tensor({269}, 0.0));
            graphs.push_back(mapped(inputs.back()));
        }
        std::vector<ormi::core::Setup> setups(workers);
        std::vector<std::exception_ptr> errors(workers);
        std::vector<std::thread> threads;
        std::latch start(1);
        for (std::size_t worker = 0; worker < workers; ++worker) {
            threads.emplace_back([&, worker] {
                start.wait();
                try {
                    setups[worker] = prepare(graphs[worker], *outputs[worker]);
                } catch (...) {
                    errors[worker] = std::current_exception();
                }
            });
        }
        start.count_down();
        for (auto& thread : threads) {
            thread.join();
        }
        for (const auto& error : errors) {
            if (error) {
                std::rethrow_exception(error);
            }
        }

        CUfunction shared = nullptr;
        for (std::size_t worker = 0; worker < workers; ++worker) {
            const auto graph = capture(setups[worker], *outputs[worker]);
            const auto nodes = kernels(graph);
            ASSERT_EQ(nodes.size(), 1u);
            if (!shared) {
                shared = nodes[0].func;
            }
            EXPECT_EQ(nodes[0].func, shared);
            setups[worker].run(*outputs[worker]);
            const auto value = 0.5 + static_cast<double>(worker) * 0.125;
            EXPECT_DOUBLE_EQ(read(*outputs[worker]), value * value + value);
        }
    }

    TEST_P(CUDAFusionExecutionTest, keepsCompiledFunctionsOnTheirCUDADevice) {
        int devices = 0;
        ASSERT_EQ(cudaGetDeviceCount(&devices), cudaSuccess);
        if (devices < 2) {
            GTEST_SKIP() << "At least two CUDA devices are required.";
        }
        const auto first_device = Device::from("cuda:0");
        const auto second_device = Device::from("cuda:1");
        const auto first_input = tensor({271}, 0.5, first_device);
        const auto second_input = tensor({271}, 0.25, second_device);
        const auto first_output = tensor({271}, 0.0, first_device);
        const auto second_output = tensor({271}, 0.0, second_device);
        auto first = prepare(mapped(first_input), *first_output);
        auto second = prepare(mapped(second_input), *second_output);
        const auto first_graph = capture(first, *first_output);
        const auto second_graph = capture(second, *second_output);
        const auto first_nodes = kernels(first_graph);
        const auto second_nodes = kernels(second_graph);
        ASSERT_EQ(first_nodes.size(), 1u);
        ASSERT_EQ(second_nodes.size(), 1u);
        EXPECT_NE(first_nodes[0].func, second_nodes[0].func);

        first.run(*first_output);
        second.run(*second_output);
        EXPECT_DOUBLE_EQ(read(*first_output), 0.75);
        EXPECT_DOUBLE_EQ(read(*second_output), 0.3125);
    }

    TEST_P(CUDAFusionExecutionTest, smallSumUsesDirectOutputExceptForFloat16) {
        const auto input = tensor({37});
        const auto output = tensor({}, 0.0);
        auto setup = prepare(ops::sum(mapped(input)), *output);
        const auto captured = capture(setup, *output);
        const auto nodes = kernels(captured);
        const bool partial = GetParam() == DType::float16;
        ASSERT_EQ(nodes.size(), partial ? 2 : 1);
        EXPECT_EQ(setup.buffers(), partial ? 1 : 0);
        EXPECT_EQ(setup.bytes(), partial ? accumulator_bytes() : 0);
        for (const auto& node : nodes) {
            expect_dimensions(node);
            EXPECT_EQ(node.gridDimX, 1u);
            if (partial && *static_cast<void**>(node.kernelParams[0]) == output->data()) {
                EXPECT_EQ(node.blockDimX, 1u);
                EXPECT_EQ(node.sharedMemBytes, accumulator_bytes());
                record_launch("finish", node, 2);
            } else {
                EXPECT_EQ(node.blockDimX, std::min(available_threads(node.func), 64u));
                EXPECT_EQ(node.sharedMemBytes, node.blockDimX * accumulator_bytes());
                EXPECT_EQ(*static_cast<const void**>(node.kernelParams[1]), input->data());
                EXPECT_EQ(*static_cast<unsigned long long*>(node.kernelParams[2]), 1ull);
                record_launch("main", node, 2);
            }
        }
        setup.run(*output);
        EXPECT_EQ(read(*output), 27.75);
    }

    TEST_P(CUDAFusionExecutionTest, singlePartMeanStillRunsFinishKernel) {
        const auto input = tensor({37});
        const auto output = tensor({}, 0.0);
        auto setup = prepare(ops::mean(mapped(input)), *output);
        const auto captured = capture(setup, *output);
        const auto nodes = kernels(captured);
        ASSERT_EQ(nodes.size(), 2);
        EXPECT_EQ(setup.buffers(), 1);
        EXPECT_EQ(setup.bytes(), accumulator_bytes());
        for (const auto& node : nodes) {
            expect_dimensions(node);
            EXPECT_EQ(node.gridDimX, 1u);
            EXPECT_EQ(*static_cast<unsigned long long*>(node.kernelParams[2]), 1ull);
            if (*static_cast<void**>(node.kernelParams[0]) == output->data()) {
                EXPECT_EQ(node.blockDimX, 1u);
                EXPECT_EQ(node.sharedMemBytes, accumulator_bytes());
                record_launch("finish", node, 2);
            } else {
                EXPECT_EQ(node.blockDimX, std::min(available_threads(node.func), 64u));
                EXPECT_EQ(node.sharedMemBytes, node.blockDimX * accumulator_bytes());
                EXPECT_EQ(*static_cast<const void**>(node.kernelParams[1]), input->data());
                record_launch("main", node, 2);
            }
        }
        setup.run(*output);
        EXPECT_NEAR(read(*output), 0.75, 1.0e-6);
    }

    TEST_P(CUDAFusionExecutionTest, groupedLaunchBindsAuxiliaryOutputsBeforeInputs) {
        const auto lhs = tensor({257});
        const auto rhs = tensor({257}, 0.25);
        const auto square = ops::mul(lhs, lhs);
        const auto graph = ops::add(square, rhs);
        Tensor* graph_outputs[]{square.get()};
        const auto expression = runtime::fusion::build(*graph, graph_outputs);
        ASSERT_TRUE(expression);
        const auto output = tensor({257}, 0.0);
        const auto auxiliary = tensor({257}, 0.0);
        Tensor* outputs[]{auxiliary.get()};
        auto setup = runtime::backend(output->device()).fusion.prepare(
            *output, std::span<Tensor* const>{outputs}, *expression
        );
        const auto captured = capture(setup, *output);
        const auto nodes = kernels(captured);
        ASSERT_EQ(nodes.size(), 1);
        const auto& main = nodes[0];
        expect_dimensions(main);
        EXPECT_EQ(setup.buffers(), 0);
        EXPECT_EQ(setup.bytes(), 0);
        EXPECT_EQ(*static_cast<void**>(main.kernelParams[0]), output->data());
        EXPECT_EQ(*static_cast<void**>(main.kernelParams[1]), auxiliary->data());
        EXPECT_EQ(*static_cast<const void**>(main.kernelParams[2]), lhs->data());
        EXPECT_EQ(*static_cast<const void**>(main.kernelParams[3]), rhs->data());
        EXPECT_EQ(*static_cast<unsigned long long*>(main.kernelParams[4]), 257ull);
        record_launch("main", main, 4);
        setup.run(*output);
        EXPECT_EQ(read(*output), 0.5);
        EXPECT_EQ(read(*auxiliary), 0.25);
    }

    TEST_P(CUDAFusionExecutionTest, splitReductionBindsWorkspaceAndFinishCount) {
        for (const auto elements : {4097, 524289}) {
            const auto input = tensor({elements});
            const auto output = tensor({}, 0.0);
            auto setup = prepare(ops::mean(mapped(input)), *output);
            const auto captured = capture(setup, *output);
            const auto nodes = kernels(captured);
            ASSERT_EQ(nodes.size(), 2);
            const auto main = std::find_if(nodes.begin(), nodes.end(), [&](const auto& node) {
                return *static_cast<void**>(node.kernelParams[0]) != output->data();
            });
            ASSERT_NE(main, nodes.end());
            const auto finish = main == nodes.begin() ? nodes.begin() + 1 : nodes.begin();
            const auto parts = (elements + main->blockDimX * 8 - 1) / (main->blockDimX * 8);
            expect_dimensions(*main);
            expect_dimensions(*finish);
            EXPECT_EQ(main->blockDimX, std::min(available_threads(main->func), 256u));
            EXPECT_EQ(main->gridDimX, parts);
            EXPECT_EQ(main->sharedMemBytes, main->blockDimX * accumulator_bytes());
            EXPECT_EQ(finish->gridDimX, 1u);
            EXPECT_EQ(finish->blockDimX, rounded_threads(parts));
            EXPECT_EQ(finish->sharedMemBytes, finish->blockDimX * accumulator_bytes());
            EXPECT_EQ(setup.buffers(), 1);
            EXPECT_EQ(setup.bytes(), parts * accumulator_bytes());
            EXPECT_EQ(*static_cast<const void**>(main->kernelParams[1]), input->data());
            EXPECT_EQ(*static_cast<void**>(finish->kernelParams[0]), output->data());
            EXPECT_EQ(*static_cast<void**>(finish->kernelParams[1]),
                *static_cast<void**>(main->kernelParams[0]));
            EXPECT_EQ(*static_cast<unsigned long long*>(main->kernelParams[2]), parts);
            EXPECT_EQ(*static_cast<unsigned long long*>(finish->kernelParams[2]), parts);
            record_launch("main_" + std::to_string(elements), *main, 2);
            record_launch("finish_" + std::to_string(elements), *finish, 2);
            setup.run(*output);
            EXPECT_NEAR(read(*output), 0.75, 1.0e-6);
        }
    }

    TEST_P(CUDAFusionExecutionTest, tiledReductionHasNoWorkspaceOrFinishLaunch) {
        const auto input = tensor({17, 1024});
        const auto graph = ops::mean(mapped(input), {0});
        const auto expression = runtime::fusion::build(*graph);
        ASSERT_TRUE(expression);
        const auto tile = ormi::backends::cuda::fusion::tile(*expression);
        ASSERT_TRUE(tile);
        const auto output = tensor({1024}, 0.0);
        auto setup = prepare(graph, *output);
        const auto captured = capture(setup, *output);
        const auto nodes = kernels(captured);
        ASSERT_EQ(nodes.size(), 1);
        const auto& main = nodes[0];
        expect_dimensions(main);
        EXPECT_EQ(main.gridDimX, std::min(tile->tasks, std::size_t{65535}));
        EXPECT_EQ(main.blockDimX, tile->width);
        EXPECT_EQ(main.sharedMemBytes, 0u);
        EXPECT_EQ(setup.buffers(), 0);
        EXPECT_EQ(setup.bytes(), 0);
        EXPECT_EQ(*static_cast<void**>(main.kernelParams[0]), output->data());
        EXPECT_EQ(*static_cast<const void**>(main.kernelParams[1]), input->data());
        EXPECT_EQ(*static_cast<unsigned long long*>(main.kernelParams[2]), 1ull);
        record_launch("main", main, 2);
        setup.run(*output);
        EXPECT_NEAR(read(*output), 0.75, 1.0e-6);
    }

    TEST_P(CUDAFusionExecutionTest, movedSetupKeepsBindingsAndRefreshesOutputPointer) {
        const auto input = tensor({4097});
        const auto original = tensor({}, 0.0);
        const auto replacement = tensor({}, 0.0);
        std::vector<ormi::core::Setup> setups;
        setups.push_back(prepare(ops::mean(mapped(input)), *original));
        setups.reserve(32);
        ormi::core::Setup moved = std::move(setups[0]);
        EXPECT_FALSE(setups[0]);
        EXPECT_EQ(moved.buffers(), 1);
        moved.run(*replacement);
        EXPECT_NEAR(read(*replacement), 0.75, 1.0e-6);
        EXPECT_EQ(read(*original), 0.0);
        const auto captured = capture(moved, *original);
        EXPECT_EQ(kernels(captured).size(), 2);
        moved.run(*original);
        EXPECT_NEAR(read(*original), 0.75, 1.0e-6);
    }

    TEST_P(CUDAFusionExecutionTest, moveAssignmentReleasesOldWorkspaceWithQueuedWork) {
        const auto first = tensor({4097});
        const auto second = tensor({524289}, 0.25);
        const auto output = tensor({}, 0.0);
        auto setup = prepare(ops::mean(mapped(first)), *output);
        setup.run(*output);
        auto replacement = prepare(ops::mean(mapped(second)), *output);
        const auto bytes = replacement.bytes();
        setup = std::move(replacement);
        EXPECT_FALSE(replacement);
        EXPECT_EQ(setup.bytes(), bytes);
        for (int repetition = 0; repetition < 8; ++repetition) {
            setup.run(*output);
        }
        setup = {};
        EXPECT_NEAR(read(*output), 0.3125, 1.0e-6);
    }

    INSTANTIATE_TEST_SUITE_P(FloatingPoint, CUDAFusionExecutionTest,
        ::testing::Values(DType::float16, DType::float32, DType::float64),
        [](const auto& parameter) { return std::string(to_string(parameter.param)); }
    );
}
