#include <cmath>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/memory/ops.h>
#include <ormi/core/op.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/optim/sgd.h>
#include <ormi/core/plan.h>
#include <ormi/core/tensor.h>

#include "core/runtime/dispatch/device.h"
#include "core/runtime/device.h"

namespace {
    using Tensor = std::shared_ptr<ormi::core::Tensor>;
    using Plan = ormi::core::Plan;
    using Update = ormi::core::Plan::Update;

    bool fail_copy = false;

    void copy_or_throw(ormi::core::Tensor& out) {
        ormi::core::runtime::backend(out.device()).memory.copy_async(
            out.device(), out.data(), out.parents()[0]->data(), out.bytes()
        );
        if (fail_copy) {
            ORMI_THROW(ormi::OperationError, "Test execution failed.");
        }
    }

    const ormi::core::Op copy_op{
        .name = "recording_test_copy",
        .output = ormi::core::Output::allocate,
        .parents = 1,
        .compute = copy_or_throw,
        .adjoint = nullptr,
        .scalar = std::monostate{}
    };

    Tensor copy_tensor(const Tensor& input) {
        return std::make_shared<ormi::core::Tensor>(
            ormi::core::op_ref<copy_op>(), std::vector<Tensor>{input},
            input->device(), input->shape(), input->dtype()
        );
    }

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
        EXPECT_EQ(plan.storage_buffers(), 2u);
        EXPECT_EQ(plan.storage_bytes(), 2u * sizeof(float));

        plan.run();
        ormi::core::runtime::device::synchronize(input->device());
        EXPECT_FLOAT_EQ(read(negative), -1.0f);
        EXPECT_FLOAT_EQ(read(exponential), std::exp(1.0f));

        const auto negative_buffer = negative->buffer();
        const auto exponential_buffer = exponential->buffer();
        constexpr float next = 0.75f;
        ormi::core::memory::copy(
            input->data(), input->device(), &next,
            ormi::core::Device::cpu(), sizeof(next)
        );
        plan.run();
        ormi::core::runtime::device::synchronize(input->device());
        EXPECT_EQ(negative->buffer(), negative_buffer);
        EXPECT_EQ(exponential->buffer(), exponential_buffer);
        EXPECT_FLOAT_EQ(read(negative), -1.5f);
        EXPECT_FLOAT_EQ(read(exponential), std::exp(1.5f));
    }

    TEST_F(CUDAPlanTest, fusesGroupedFloat64Outputs) {
        const auto input = scalar(0.25);
        const auto mapped = ormi::core::ops::add(input, input);
        const auto negative = ormi::core::ops::neg(mapped);
        const auto exponential = ormi::core::ops::exp(mapped);
        Plan plan({negative, exponential});

        EXPECT_EQ(plan.steps(), 1u);
        EXPECT_EQ(plan.storage_buffers(), 2u);
        EXPECT_EQ(plan.storage_bytes(), 2u * sizeof(double));

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
        EXPECT_EQ(plan.storage_buffers(), 4u);
        EXPECT_EQ(plan.storage_bytes(), 4u * sizeof(Float16));

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
        EXPECT_EQ(plan.storage_buffers(), 0u);
        EXPECT_EQ(plan.storage_bytes(), 0u);

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

    TEST_F(CUDAPlanTest, ownsGraphUntilReplayIsReleased) {
        std::weak_ptr<ormi::core::Tensor> input_owner;
        std::weak_ptr<ormi::core::Tensor> output_owner;
        std::optional<Plan::Replay> replay;
        {
            const auto input = scalar(0.5f);
            const auto output = ormi::core::ops::neg(input);
            input_owner = input;
            output_owner = output;
            Plan plan({output});
            plan.run();
            ormi::core::runtime::device::synchronize(input->device());
            replay.emplace(plan.record());
        }

        ASSERT_FALSE(input_owner.expired());
        ASSERT_FALSE(output_owner.expired());
        replay->run();
        EXPECT_FLOAT_EQ(read(output_owner.lock()), -0.5f);

        replay.reset();
        EXPECT_TRUE(input_owner.expired());
        EXPECT_TRUE(output_owner.expired());
    }

    TEST_F(CUDAPlanTest, moveAssignmentReleasesPreviousRecording) {
        std::weak_ptr<ormi::core::Tensor> first_owner;
        std::weak_ptr<ormi::core::Tensor> second_owner;
        std::optional<Plan::Replay> first;
        std::optional<Plan::Replay> second;
        {
            const auto input = scalar(1.0f);
            const auto output = ormi::core::ops::neg(input);
            first_owner = output;
            Plan plan({output});
            plan.run();
            ormi::core::runtime::device::synchronize(input->device());
            first.emplace(plan.record());
        }
        {
            const auto input = scalar(2.0f);
            const auto output = ormi::core::ops::neg(input);
            second_owner = output;
            Plan plan({output});
            plan.run();
            ormi::core::runtime::device::synchronize(input->device());
            second.emplace(plan.record());
        }

        *first = std::move(*second);
        second.reset();
        EXPECT_TRUE(first_owner.expired());
        ASSERT_FALSE(second_owner.expired());
        first->run();
        EXPECT_FLOAT_EQ(read(second_owner.lock()), -2.0f);

        first.reset();
        EXPECT_TRUE(second_owner.expired());
    }

    TEST_F(CUDAPlanTest, releasesRecordingWithQueuedReplays) {
        const auto device = ormi::core::Device::from("cuda");
        std::weak_ptr<ormi::core::Tensor> input_owner;
        std::weak_ptr<ormi::core::Tensor> output_owner;
        {
            const auto input = ormi::core::ops::full({1048576}, 0.5f, device);
            input->realize();
            const auto output = ormi::core::ops::neg(input);
            input_owner = input;
            output_owner = output;
            Plan plan({output});
            plan.run();
            ormi::core::runtime::device::synchronize(device);
            auto replay = plan.record();
            for (int run = 0; run < 32; ++run) {
                replay.run();
            }
        }

        EXPECT_TRUE(input_owner.expired());
        EXPECT_TRUE(output_owner.expired());
        EXPECT_NO_THROW(ormi::core::runtime::device::synchronize(device));
        EXPECT_FLOAT_EQ(read(scalar(3.0f)), 3.0f);
    }

    TEST_F(CUDAPlanTest, rejectsRecordingAcrossDevices) {
        const auto cpu_input = ormi::core::ops::full(
            {}, 1.0f, ormi::core::Device::cpu()
        );
        cpu_input->realize();
        const auto cuda_input = scalar(2.0f);
        const auto cpu_output = ormi::core::ops::neg(cpu_input);
        const auto cuda_output = ormi::core::ops::neg(cuda_input);
        Plan plan({cpu_output, cuda_output});
        plan.run();
        ormi::core::runtime::device::synchronize(cuda_input->device());

        EXPECT_THROW(static_cast<void>(plan.record()), ormi::OperationError);
        plan.run();
        EXPECT_FLOAT_EQ(*static_cast<const float*>(cpu_output->data()), -1.0f);
        EXPECT_FLOAT_EQ(read(cuda_output), -2.0f);
    }

    TEST_F(CUDAPlanTest, recoversAfterRecordingBodyThrows) {
        const auto input = scalar(2.0f);
        const auto output = copy_tensor(input);
        Plan plan({output});
        plan.run();
        ormi::core::runtime::device::synchronize(input->device());
        const auto buffer = output->buffer();

        fail_copy = true;
        EXPECT_THROW(static_cast<void>(plan.record()), ormi::OperationError);
        fail_copy = false;
        EXPECT_EQ(output->buffer(), buffer);

        plan.run();
        EXPECT_FLOAT_EQ(read(output), 2.0f);
        auto replay = plan.record();
        replay.run();
        EXPECT_FLOAT_EQ(read(output), 2.0f);
    }

    TEST_F(CUDAPlanTest, rejectsRecordingAfterFailedFirstRun) {
        const auto input = scalar(2.0f);
        const auto output = copy_tensor(input);
        Plan plan({output});

        fail_copy = true;
        EXPECT_THROW(plan.run(), ormi::OperationError);
        fail_copy = false;
        EXPECT_THROW(static_cast<void>(plan.record()), ormi::OperationError);

        plan.run();
        EXPECT_FLOAT_EQ(read(output), 2.0f);
        auto replay = plan.record();
        replay.run();
        EXPECT_FLOAT_EQ(read(output), 2.0f);
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
