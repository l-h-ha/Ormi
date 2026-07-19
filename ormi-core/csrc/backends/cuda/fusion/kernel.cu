#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>

#include <ormi/core/error.h>
#include <ormi/core/tensor.h>

#include "backends/cuda/check.cuh"
#include "backends/cuda/fusion/codegen.h"
#include "backends/cuda/fusion/kernel.h"
#include "backends/cuda/state.h"

namespace ormi::backends::cuda::fusion {
    namespace {
        void check_driver(CUresult status, const char* operation) {
            if (status == CUDA_SUCCESS) {
                return;
            }

            const char* name = nullptr;
            const char* description = nullptr;
            (void)cuGetErrorName(status, &name);
            (void)cuGetErrorString(status, &description);
            ORMI_THROW(
                cudaError,
                std::string(operation) + " failed with " +
                    (name ? name : "unknown CUDA driver error") + ": " +
                    (description ? description : "no description available")
            );
        }

        class Program final {
        public:
            explicit Program(const std::string& source) {
                const auto status = nvrtcCreateProgram(
                    &program_,
                    source.c_str(),
                    "ormi_fusion.cu",
                    0,
                    nullptr,
                    nullptr
                );

                if (status != NVRTC_SUCCESS) {
                    ORMI_THROW(
                        jitError,
                        std::string("nvrtcCreateProgram failed: ") + nvrtcGetErrorString(status)
                    );
                }
            }

            ~Program() noexcept {
                if (program_) {
                    (void)nvrtcDestroyProgram(&program_);
                }
            }

            Program(const Program&) = delete;
            Program& operator=(const Program&) = delete;

            std::string compile(int major, int minor) {
                const auto architecture =
                    "--gpu-architecture=compute_" + std::to_string(major) + std::to_string(minor);
                const char* options[]{"--std=c++17", architecture.c_str()};
                const auto status = nvrtcCompileProgram(program_, 2, options);

                std::size_t log_size = 0;
                (void)nvrtcGetProgramLogSize(program_, &log_size);
                std::string log(log_size, '\0');
                if (log_size != 0) {
                    (void)nvrtcGetProgramLog(program_, log.data());
                }

                if (status != NVRTC_SUCCESS) {
                    ORMI_THROW(
                        jitError,
                        std::string("CUDA fusion compilation failed: ") +
                            nvrtcGetErrorString(status) + "\n" + log
                    );
                }

                std::size_t ptx_size = 0;
                if (nvrtcGetPTXSize(program_, &ptx_size) != NVRTC_SUCCESS) {
                    ORMI_THROW(jitError, "Cannot determine CUDA fusion PTX size.");
                }

                std::string ptx(ptx_size, '\0');
                if (nvrtcGetPTX(program_, ptx.data()) != NVRTC_SUCCESS) {
                    ORMI_THROW(jitError, "Cannot retrieve CUDA fusion PTX.");
                }
                return ptx;
            }

        private:
            nvrtcProgram program_ = nullptr;
        };

        struct Compiled {
            CUmodule module = nullptr;
            CUfunction function = nullptr;
            int threads = 0;
            int max_blocks = 0;

            ~Compiled() noexcept {
                if (module) {
                    (void)cuModuleUnload(module);
                }
            }
        };

        std::uint64_t combine(std::uint64_t hash, std::uint64_t value) {
            hash ^= value;
            hash *= 1099511628211ull;
            return hash;
        }

        std::uint64_t structural_hash(const core::runtime::fusion::Expr& expression) {
            auto result = combine(
                14695981039346656037ull,
                static_cast<std::uint64_t>(expression.dtype)
            );
            result = combine(result, expression.output);
            result = combine(result, expression.nodes.size());

            for (const auto& node : expression.nodes) {
                result = combine(result, node.index());
                std::visit(
                    [&result](const auto& value) {
                        using T = std::decay_t<decltype(value)>;
                        if constexpr (std::is_same_v<T, core::runtime::fusion::Input>) {
                            result = combine(result, value.index);
                        } else if constexpr (std::is_same_v<T, core::runtime::fusion::Constant>) {
                            result = combine(result, static_cast<std::uint64_t>(value));
                        } else if constexpr (std::is_same_v<T, core::runtime::fusion::Unary>) {
                            result = combine(result, static_cast<std::uint64_t>(value.op));
                            result = combine(result, value.input);
                        } else {
                            result = combine(result, static_cast<std::uint64_t>(value.op));
                            result = combine(result, value.lhs);
                            result = combine(result, value.rhs);
                        }
                    },
                    node
                );
            }

            return result;
        }

        struct Entry {
            int device;
            int major;
            int minor;
            core::DType dtype;
            std::vector<core::runtime::fusion::Node> nodes;
            core::runtime::fusion::Index output;
            std::shared_ptr<Compiled> compiled;

            bool matches(
                int requested_device,
                int requested_major,
                int requested_minor,
                const core::runtime::fusion::Expr& expression
            ) const {
                return
                    device == requested_device &&
                    major == requested_major &&
                    minor == requested_minor &&
                    dtype == expression.dtype &&
                    output == expression.output &&
                    nodes == expression.nodes;
            }
        };

        std::shared_ptr<Compiled> compile(
            const core::runtime::fusion::Expr& expression,
            const State& cuda_state,
            int major,
            int minor
        ) {
            Program program(emit(expression));
            const auto ptx = program.compile(major, minor);
            auto result = std::make_shared<Compiled>();

            check_driver(cuModuleLoadData(&result->module, ptx.data()), "cuModuleLoadData");

            try {
                check_driver(
                    cuModuleGetFunction(&result->function, result->module, "ormi_kernel"),
                    "cuModuleGetFunction"
                );

                int minimum_blocks = 0;
                check_driver(
                    cuOccupancyMaxPotentialBlockSize(
                        &minimum_blocks,
                        &result->threads,
                        result->function,
                        nullptr,
                        0,
                        0
                    ),
                    "cuOccupancyMaxPotentialBlockSize"
                );

                int active_blocks = 0;
                check_driver(
                    cuOccupancyMaxActiveBlocksPerMultiprocessor(
                        &active_blocks,
                        result->function,
                        result->threads,
                        0
                    ),
                    "cuOccupancyMaxActiveBlocksPerMultiprocessor"
                );

                result->max_blocks = std::max(
                    1,
                    cuda_state.multiprocessors() * active_blocks
                );
            } catch (...) {
                (void)cuModuleUnload(result->module);
                result->module = nullptr;
                throw;
            }

            return result;
        }

        std::shared_ptr<Compiled> kernel(
            const core::runtime::fusion::Expr& expression,
            const State& cuda_state,
            int major,
            int minor
        ) {
            using Cache = std::unordered_map<std::uint64_t, std::vector<Entry>>;
            static std::mutex mutex;
            static std::atomic<std::shared_ptr<const Cache>> published{
                std::make_shared<const Cache>()
            };
            const int device = cuda_state.index();
            auto hash = structural_hash(expression);
            hash = combine(hash, static_cast<std::uint64_t>(device));
            hash = combine(hash, static_cast<std::uint64_t>(major));
            hash = combine(hash, static_cast<std::uint64_t>(minor));

            auto current = published.load(std::memory_order_acquire);
            if (const auto found = current->find(hash); found != current->end()) {
                for (const auto& entry : found->second) {
                    if (entry.matches(device, major, minor, expression)) {
                        return entry.compiled;
                    }
                }
            }

            std::lock_guard lock(mutex);
            current = published.load(std::memory_order_acquire);
            if (const auto found = current->find(hash); found != current->end()) {
                for (const auto& entry : found->second) {
                    if (entry.matches(device, major, minor, expression)) {
                        return entry.compiled;
                    }
                }
            }

            auto compiled = compile(expression, cuda_state, major, minor);
            auto next = std::make_shared<Cache>(*current);
            (*next)[hash].push_back(Entry{
                device,
                major,
                minor,
                expression.dtype,
                expression.nodes,
                expression.output,
                compiled
            });
            published.store(std::move(next), std::memory_order_release);
            return compiled;
        }
    }

    void execute(core::Tensor& out, const core::runtime::fusion::Expr& expression) {
        if (out.numel() == 0) {
            return;
        }

        const int device = out.device().index();
        auto& cuda_state = state(out.device());
        check(cudaSetDevice(device), "cudaSetDeviceFusion");

        auto compiled = kernel(
            expression,
            cuda_state,
            cuda_state.compute_major(),
            cuda_state.compute_minor()
        );
        void* output = out.data();
        thread_local std::vector<const void*> inputs;
        inputs.clear();
        inputs.reserve(expression.inputs.size());
        for (const auto* input : expression.inputs) {
            inputs.push_back(input->data());
        }

        auto elements = static_cast<unsigned long long>(out.numel());
        thread_local std::vector<void*> arguments;
        arguments.clear();
        arguments.reserve(inputs.size() + 2);
        arguments.push_back(&output);
        for (auto& input : inputs) {
            arguments.push_back(&input);
        }
        arguments.push_back(&elements);

        const auto required_blocks = static_cast<int>(
            std::min<unsigned long long>(
                (elements + compiled->threads - 1) / compiled->threads,
                static_cast<unsigned long long>(compiled->max_blocks)
            )
        );
        check_driver(
            cuLaunchKernel(
                compiled->function,
                required_blocks,
                1,
                1,
                compiled->threads,
                1,
                1,
                0,
                reinterpret_cast<CUstream>(cuda_state.stream()),
                arguments.data(),
                nullptr
            ),
            "cuLaunchKernel"
        );
    }
}
