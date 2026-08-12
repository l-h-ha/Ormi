#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <latch>
#include <memory>
#include <thread>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/memory/ops.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/plan.h>
#include <ormi/core/tensor.h>

#include "core/runtime/dispatch/device.h"

namespace {
    using Tensor = std::shared_ptr<ormi::core::Tensor>;

    class CUDAFusionLifetimeTest : public ::testing::Test {
    protected:
        void SetUp() override {
            int devices = 0;
            const auto status = cudaGetDeviceCount(&devices);
            if (status != cudaSuccess) {
                GTEST_SKIP()
                    << "CUDA runtime is unavailable: "
                    << cudaGetErrorString(status);
            }
            if (devices == 0) {
                GTEST_SKIP() << "No CUDA device is available.";
            }
            ready_ = true;
        }

        void TearDown() override {
            if (!ready_) {
                return;
            }
            const auto device = ormi::core::Device::from("cuda");
            ormi::core::runtime::device::synchronize(device);
        }

        static Tensor input(std::size_t elements) {
            auto result = ormi::core::ops::full(
                {static_cast<std::int64_t>(elements)},
                0.5f,
                ormi::core::Device::from("cuda")
            );
            result->realize();
            return result;
        }

        static Tensor expression(const Tensor& value) {
            return ormi::core::ops::mean(
                ormi::core::ops::sin(
                    ormi::core::ops::add(
                        ormi::core::ops::mul(value, value),
                        value
                    )
                )
            );
        }

        static float read(const Tensor& tensor) {
            float value = 0.0f;
            ormi::core::memory::copy(
                &value,
                ormi::core::Device::cpu(),
                tensor->data(),
                tensor->device(),
                sizeof(value)
            );
            return value;
        }

    private:
        bool ready_ = false;
    };

    TEST_F(CUDAFusionLifetimeTest, releasesUnrunPreparedReductions) {
        const auto value = input(65536);
        const auto output = expression(value);

        for (int repetition = 0; repetition < 8; ++repetition) {
            const ormi::core::Plan plan({output});
            EXPECT_EQ(plan.steps(), 1);
            EXPECT_EQ(plan.storage_buffers(), 2);
            EXPECT_GT(plan.storage_bytes(), output->bytes());
        }

        EXPECT_FALSE(output->realized());
        ormi::core::runtime::device::synchronize(value->device());

        output->realize();
        EXPECT_NEAR(read(output), std::sin(0.75f), 1.0e-6f);
    }

    TEST_F(CUDAFusionLifetimeTest, matchesImmediateAndPreparedExecution) {
        const auto immediate = expression(input(4096));
        immediate->realize();
        const auto immediate_value = read(immediate);

        const auto prepared = expression(input(4096));
        ormi::core::Plan plan({prepared});
        plan.run();

        EXPECT_NEAR(read(prepared), immediate_value, 1.0e-6f);
    }

    TEST_F(CUDAFusionLifetimeTest, resolvesOneKernelAcrossConcurrentCalls) {
        constexpr std::size_t workers = 4;
        std::vector<Tensor> outputs;
        outputs.reserve(workers);
        for (std::size_t worker = 0; worker < workers; ++worker) {
            outputs.push_back(expression(input(4096)));
        }

        std::latch start(1);
        std::vector<std::exception_ptr> errors(workers);
        std::vector<std::thread> threads;
        threads.reserve(workers);
        for (std::size_t worker = 0; worker < workers; ++worker) {
            threads.emplace_back([&, worker] {
                start.wait();
                try {
                    outputs[worker]->realize();
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

        ormi::core::runtime::device::synchronize(
            ormi::core::Device::from("cuda")
        );
        for (const auto& output : outputs) {
            EXPECT_NEAR(read(output), std::sin(0.75f), 1.0e-6f);
        }
    }
}
