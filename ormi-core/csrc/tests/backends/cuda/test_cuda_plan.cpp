#include <cmath>
#include <memory>
#include <optional>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/memory/ops.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/optim/sgd.h>
#include <ormi/core/plan.h>
#include <ormi/core/tensor.h>

#include "core/runtime/dispatch/device.h"

namespace {
    using Tensor = std::shared_ptr<ormi::core::Tensor>;
    using Plan = ormi::core::Plan;
    using Update = ormi::core::Plan::Update;

    class CUDAPlanTest : public ::testing::Test {
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
        }
    };

    template <typename T>
    Tensor scalar(T value) {
        auto result = ormi::core::ops::full(
            {},
            value,
            ormi::core::Device::from("cuda")
        );
        result->realize();
        ormi::core::runtime::device::synchronize(result->device());
        return result;
    }

    template <typename T>
    T read_as(const Tensor& tensor) {
        T value{};
        ormi::core::memory::copy(
            &value,
            ormi::core::Device::cpu(),
            tensor->data(),
            tensor->device(),
            sizeof(value)
        );
        return value;
    }

    float read(const Tensor& tensor) {
        return read_as<float>(tensor);
    }

    TEST_F(CUDAPlanTest, replaysPreparedFusion) {
        const auto input = scalar(0.5f);
        const auto output = ormi::core::ops::sin(
            ormi::core::ops::add(
                ormi::core::ops::mul(input, input),
                input
            )
        );
        Plan plan({output});

        plan.run();
        ormi::core::runtime::device::synchronize(input->device());
        EXPECT_FLOAT_EQ(read(output), std::sin(0.75f));

        constexpr float next = 1.0f;
        ormi::core::memory::copy(
            input->data(),
            input->device(),
            &next,
            ormi::core::Device::cpu(),
            sizeof(next)
        );
        plan.run();
        ormi::core::runtime::device::synchronize(input->device());
        EXPECT_FLOAT_EQ(read(output), std::sin(2.0f));
    }

    TEST_F(CUDAPlanTest, fusesGroupedSiblingOutputs) {
        const auto input = scalar(0.5f);
        const auto mapped = ormi::core::ops::add(input, input);
        const auto negative = ormi::core::ops::neg(mapped);
        const auto exponential = ormi::core::ops::exp(mapped);
        Plan plan({negative, exponential});

        EXPECT_EQ(plan.steps(), 1u);

        plan.run();
        ormi::core::runtime::device::synchronize(input->device());
        EXPECT_FLOAT_EQ(read(negative), -1.0f);
        EXPECT_FLOAT_EQ(read(exponential), std::exp(1.0f));
    }

    TEST_F(CUDAPlanTest, fusesGroupedFloat64Outputs) {
        const auto input = scalar(0.25);
        const auto mapped = ormi::core::ops::add(input, input);
        const auto negative = ormi::core::ops::neg(mapped);
        const auto exponential = ormi::core::ops::exp(mapped);
        Plan plan({negative, exponential});

        EXPECT_EQ(plan.steps(), 1u);

        plan.run();
        ormi::core::runtime::device::synchronize(input->device());
        EXPECT_DOUBLE_EQ(read_as<double>(negative), -0.5);
        EXPECT_DOUBLE_EQ(read_as<double>(exponential), std::exp(0.5));
    }

    TEST_F(CUDAPlanTest, fusesFourGroupedFloat16Outputs) {
        using ormi::core::Float16;

        const auto input = scalar(Float16{0.5f});
        const auto mapped = ormi::core::ops::add(input, input);
        const auto negative = ormi::core::ops::neg(mapped);
        const auto exponential = ormi::core::ops::exp(mapped);
        const auto sine = ormi::core::ops::sin(mapped);
        const auto cosine = ormi::core::ops::cos(mapped);
        Plan plan({negative, exponential, sine, cosine});

        EXPECT_EQ(plan.steps(), 1u);

        plan.run();
        ormi::core::runtime::device::synchronize(input->device());
        EXPECT_EQ(read_as<Float16>(negative).bits(), Float16{-1.0f}.bits());
        EXPECT_EQ(
            read_as<Float16>(exponential).bits(),
            Float16{std::exp(1.0f)}.bits()
        );
        EXPECT_EQ(
            read_as<Float16>(sine).bits(),
            Float16{std::sin(1.0f)}.bits()
        );
        EXPECT_EQ(
            read_as<Float16>(cosine).bits(),
            Float16{std::cos(1.0f)}.bits()
        );
    }

    TEST_F(CUDAPlanTest, handlesEmptyGroupedOutputs) {
        const auto input = ormi::core::ops::full(
            {0},
            0.5f,
            ormi::core::Device::from("cuda")
        );
        input->realize();
        const auto mapped = ormi::core::ops::add(input, input);
        const auto negative = ormi::core::ops::neg(mapped);
        const auto exponential = ormi::core::ops::exp(mapped);
        Plan plan({negative, exponential});

        EXPECT_EQ(plan.steps(), 1u);

        plan.run();
        EXPECT_TRUE(negative->realized());
        EXPECT_TRUE(exponential->realized());
        EXPECT_EQ(negative->numel(), 0u);
        EXPECT_EQ(exponential->numel(), 0u);
    }

    TEST_F(CUDAPlanTest, ownsPreparedReductionWorkspace) {
        const auto input = ormi::core::ops::full(
            {4},
            0.5f,
            ormi::core::Device::from("cuda")
        );
        input->realize();
        const auto output = ormi::core::ops::mean(
            ormi::core::ops::sin(
                ormi::core::ops::add(
                    ormi::core::ops::mul(input, input),
                    input
                )
            )
        );
        Plan plan({output});

        EXPECT_EQ(plan.steps(), 1);
        EXPECT_EQ(plan.storage_buffers(), 2);
        EXPECT_EQ(plan.storage_bytes(), 2 * sizeof(float));

        plan.run();
        EXPECT_FLOAT_EQ(read(output), std::sin(0.75f));
    }

    TEST_F(CUDAPlanTest, recordsPreparedFusion) {
        const auto input = scalar(0.5f);
        const auto output = ormi::core::ops::sin(
            ormi::core::ops::mul(input, input)
        );
        std::optional<Plan::Replay> replay;

        {
            Plan plan({output});
            plan.run();
            ormi::core::runtime::device::synchronize(input->device());
            replay.emplace(plan.record());
        }

        replay->run();
        ormi::core::runtime::device::synchronize(input->device());
        EXPECT_FLOAT_EQ(read(output), std::sin(0.25f));
    }

    TEST_F(CUDAPlanTest, recordsPreparedGroupedFusion) {
        const auto input = scalar(0.5f);
        const auto mapped = ormi::core::ops::add(input, input);
        const auto negative = ormi::core::ops::neg(mapped);
        const auto exponential = ormi::core::ops::exp(mapped);
        Plan plan({negative, exponential});

        plan.run();
        ormi::core::runtime::device::synchronize(input->device());
        auto replay = plan.record();
        replay.run();
        ormi::core::runtime::device::synchronize(input->device());

        EXPECT_FLOAT_EQ(read(negative), -1.0f);
        EXPECT_FLOAT_EQ(read(exponential), std::exp(1.0f));
    }

    TEST_F(CUDAPlanTest, appliesUpdatesAfterEveryRun) {
        const auto parameter = scalar(2.0f);
        const auto gradient = scalar(1.0f);
        const auto next =
            ormi::core::optim::sgd(parameter, gradient, 0.1);
        Plan plan(
            {},
            std::vector<Update>{{parameter, next}}
        );

        plan.run();
        EXPECT_FLOAT_EQ(read(parameter), 1.9f);

        plan.run();
        EXPECT_FLOAT_EQ(read(parameter), 1.8f);
    }

    TEST_F(CUDAPlanTest, evaluatesAllUpdatesBeforeApplyingAny) {
        const auto first = scalar(2.0f);
        const auto second = scalar(3.0f);
        const auto next_first =
            ormi::core::ops::add(first, second);
        const auto next_second =
            ormi::core::ops::mul(first, second);
        Plan plan(
            {},
            std::vector<Update>{
                {first, next_first},
                {second, next_second}
            }
        );

        plan.run();
        EXPECT_FLOAT_EQ(read(first), 5.0f);
        EXPECT_FLOAT_EQ(read(second), 6.0f);

        plan.run();
        EXPECT_FLOAT_EQ(read(first), 11.0f);
        EXPECT_FLOAT_EQ(read(second), 30.0f);
    }

    TEST_F(CUDAPlanTest, rejectsRecordingBeforeFirstRun) {
        const auto input = scalar(2.0f);
        const auto output = ormi::core::ops::mul(input, input);
        Plan plan({output});

        EXPECT_THROW(
            static_cast<void>(plan.record()),
            ormi::OperationError
        );
    }

    TEST_F(CUDAPlanTest, recordsWarmedRun) {
        const auto parameter = scalar(2.0f);
        const auto gradient = scalar(1.0f);
        const auto next =
            ormi::core::optim::sgd(parameter, gradient, 0.1);
        std::optional<Plan::Replay> replay;
        {
            Plan plan(
                {},
                std::vector<Update>{{parameter, next}}
            );

            plan.run();
            ormi::core::runtime::device::synchronize(
                parameter->device()
            );
            ASSERT_FLOAT_EQ(read(parameter), 1.9f);

            replay.emplace(plan.record());
        }

        replay->run();
        ormi::core::runtime::device::synchronize(parameter->device());
        EXPECT_FLOAT_EQ(read(parameter), 1.8f);
    }
}
