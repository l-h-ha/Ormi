#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/op.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/plan.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace {
    using Tensor = std::shared_ptr<ormi::core::Tensor>;

    constexpr std::size_t depth = 16384;

    int shared_dependency_runs = 0;

    Tensor scalar(float value) {
        auto tensor = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{},
            ormi::core::DType::float32
        );
        *static_cast<float*>(tensor->data()) = value;
        return tensor;
    }

    std::vector<Tensor> alias_chain() {
        std::vector<Tensor> tensors;
        tensors.reserve(depth + 1);
        tensors.push_back(ormi::core::ops::full(
            {1},
            2.0f,
            ormi::core::Device::cpu()
        ));
        tensors.back()->realize();

        for (std::size_t index = 0; index < depth; ++index) {
            tensors.push_back(
                ormi::core::ops::reshape(tensors.back(), {1})
            );
        }
        return tensors;
    }

    void release(std::vector<Tensor>& tensors) {
        for (auto tensor = tensors.rbegin(); tensor != tensors.rend(); ++tensor) {
            tensor->reset();
        }
    }

    void fail(ormi::core::Tensor&) {
        ORMI_THROW(ormi::OperationError, "Scheduled operation failed.");
    }

    void count_dependency(ormi::core::Tensor& output) {
        ++shared_dependency_runs;
        *static_cast<float*>(output.data()) =
            *static_cast<const float*>(output.parents()[0]->data());
    }

    void add_parents(ormi::core::Tensor& output) {
        *static_cast<float*>(output.data()) =
            *static_cast<const float*>(output.parents()[0]->data()) +
            *static_cast<const float*>(output.parents()[1]->data());
    }

    const ormi::core::Op failing_op{
        .name = "failing",
        .output = ormi::core::Output::allocate,
        .parents = 1,
        .compute = fail,
        .adjoint = nullptr,
        .scalar = std::monostate{}
    };

    const ormi::core::Op shared_dependency_op{
        .name = "shared_dependency",
        .output = ormi::core::Output::allocate,
        .parents = 1,
        .compute = count_dependency,
        .adjoint = nullptr,
        .scalar = std::monostate{}
    };

    const ormi::core::Op add_parents_op{
        .name = "add_parents",
        .output = ormi::core::Output::allocate,
        .parents = 2,
        .compute = add_parents,
        .adjoint = nullptr,
        .scalar = std::monostate{}
    };

    TEST(ScheduleTest, realizesDeepAliasChain) {
        auto tensors = alias_chain();

        tensors.back()->realize();

        EXPECT_TRUE(tensors.back()->realized());
        EXPECT_EQ(tensors.back()->buffer(), tensors.front()->buffer());
        release(tensors);
    }

    TEST(ScheduleTest, preparesDeepAliasChain) {
        auto tensors = alias_chain();

        {
            ormi::core::Plan plan({tensors.back()});
            EXPECT_EQ(plan.steps(), depth);
            plan.run();
            EXPECT_TRUE(tensors.back()->realized());
            EXPECT_EQ(tensors.back()->buffer(), tensors.front()->buffer());
        }

        release(tensors);
    }

    TEST(ScheduleTest, executesSharedDependencyOnce) {
        shared_dependency_runs = 0;
        const auto input = ormi::core::ops::full(
            {},
            3.0f,
            ormi::core::Device::cpu()
        );
        const auto shared = std::make_shared<ormi::core::Tensor>(
            ormi::core::op_ref<shared_dependency_op>(),
            std::vector<Tensor>{input},
            input->device(),
            input->shape(),
            input->dtype()
        );
        const auto output = std::make_shared<ormi::core::Tensor>(
            ormi::core::op_ref<add_parents_op>(),
            std::vector<Tensor>{shared, shared},
            shared->device(),
            shared->shape(),
            shared->dtype()
        );

        output->realize();

        EXPECT_EQ(shared_dependency_runs, 1);
        EXPECT_FLOAT_EQ(*static_cast<const float*>(output->data()), 6.0f);
    }

    TEST(ScheduleTest, realizesFusedExpression) {
        const auto input = ormi::core::ops::full(
            {},
            2.0f,
            ormi::core::Device::cpu()
        );
        const auto output = ormi::core::ops::add(
            ormi::core::ops::mul(input, input),
            input
        );

        output->realize();

        EXPECT_FLOAT_EQ(*static_cast<const float*>(output->data()), 6.0f);
    }

    TEST(ScheduleTest, realizesFusedExpressionWithReadyInput) {
        const auto input = scalar(2.0f);
        const auto output = ormi::core::ops::add(
            ormi::core::ops::mul(input, input),
            input
        );

        output->realize();

        EXPECT_FLOAT_EQ(*static_cast<const float*>(output->data()), 6.0f);
    }

    TEST(ScheduleTest, realizesGroupedDependenciesBelowDirectOperation) {
        const auto input = ormi::core::ops::full(
            {},
            2.0f,
            ormi::core::Device::cpu()
        );
        const auto mapped = ormi::core::ops::add(input, input);
        const auto negative = ormi::core::ops::neg(mapped);
        const auto exponential = ormi::core::ops::exp(mapped);
        const auto output = std::make_shared<ormi::core::Tensor>(
            ormi::core::op_ref<add_parents_op>(),
            std::vector<Tensor>{negative, exponential},
            input->device(),
            input->shape(),
            input->dtype()
        );

        output->realize();

        EXPECT_FLOAT_EQ(
            *static_cast<const float*>(output->data()),
            std::exp(4.0f) - 4.0f
        );
    }

    TEST(ScheduleTest, leavesFailedOutputLazy) {
        const auto input = ormi::core::ops::full(
            {1},
            2.0f,
            ormi::core::Device::cpu()
        );
        const auto output = std::make_shared<ormi::core::Tensor>(
            ormi::core::op_ref<failing_op>(),
            std::vector<Tensor>{input},
            input->device(),
            input->shape(),
            input->dtype()
        );

        EXPECT_THROW(output->realize(), ormi::OperationError);
        EXPECT_TRUE(input->realized());
        EXPECT_FALSE(output->realized());
    }

    TEST(ScheduleTest, leavesFailedReadyOutputLazy) {
        const auto input = scalar(2.0f);
        const auto output = std::make_shared<ormi::core::Tensor>(
            ormi::core::op_ref<failing_op>(),
            std::vector<Tensor>{input},
            input->device(),
            input->shape(),
            input->dtype()
        );

        EXPECT_THROW(output->realize(), ormi::OperationError);
        EXPECT_FALSE(output->realized());
    }
}
