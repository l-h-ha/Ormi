#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/autograd.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/linalg.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/plan.h>

namespace {
    using namespace ormi::core;

    TEST(PartialTest, differentiatesInteractingPointsWithoutSummingOutputs) {
        const auto input = std::make_shared<Tensor>(Device::cpu(), Shape{2, 1}, DType::float64);
        auto* values = static_cast<double*>(input->data());
        values[0] = 2.0;
        values[1] = 4.0;
        const auto output = ops::sub(input, ops::mean(input, {0}, true));
        const auto first = partial(output, input);
        const auto second = partial(output, input, {}, {}, 2);
        const auto summed = grad(ops::sum(output), input);
        EXPECT_FALSE(output->realized());
        EXPECT_FALSE(first->realized());
        EXPECT_FALSE(second->realized());
        EXPECT_EQ(first->shape(), (Shape{2, 1}));

        Plan({first, second, summed}).run();
        for (std::size_t i = 0; i < 2; ++i) {
            EXPECT_DOUBLE_EQ(static_cast<const double*>(first->data())[i], 0.5);
            EXPECT_DOUBLE_EQ(static_cast<const double*>(second->data())[i], 0.0);
            EXPECT_DOUBLE_EQ(static_cast<const double*>(summed->data())[i], 0.0);
        }
    }

    TEST(PartialTest, preservesAllPointDimensionsAndSelectsOutputComponents) {
        const auto input = std::make_shared<Tensor>(Device::cpu(), Shape{2, 1, 2, 2}, DType::float64);
        std::fill_n(static_cast<double*>(input->data()), input->numel(), 2.0);
        const auto weight = std::make_shared<Tensor>(Device::cpu(), Shape{2, 2}, DType::float64);
        const std::array values{1.0, 2.0, 3.0, 4.0};
        std::copy(values.begin(), values.end(), static_cast<double*>(weight->data()));
        const auto output = ops::matmul(ops::mul(input, input), weight);
        const auto first = partial(output, input, -1, 0);
        const auto second = partial(output, input, -1, 0, 2);
        const auto weight_gradient = grad(ops::sum(second), weight);
        EXPECT_EQ(first->shape(), (Shape{2, 1, 2, 1}));
        EXPECT_FALSE(weight_gradient->realized());
        Plan({first, second, weight_gradient}).run();
        for (std::size_t i = 0; i < 4; ++i) {
            EXPECT_DOUBLE_EQ(static_cast<const double*>(first->data())[i], 12.0);
            EXPECT_DOUBLE_EQ(static_cast<const double*>(second->data())[i], 6.0);
        }
        const auto* derivative = static_cast<const double*>(weight_gradient->data());
        EXPECT_DOUBLE_EQ(derivative[0], 0.0);
        EXPECT_DOUBLE_EQ(derivative[1], 0.0);
        EXPECT_DOUBLE_EQ(derivative[2], 8.0);
        EXPECT_DOUBLE_EQ(derivative[3], 0.0);
    }

    TEST(PartialTest, rejectsAmbiguousSelectorsAndInvalidMetadataBeforeExecution) {
        const auto input = ops::full({2, 3}, 1.0, Device::cpu());
        const auto output = ops::full({2, 2}, 1.0, Device::cpu());
        EXPECT_THROW(partial(nullptr, input), ormi::OperationError);
        EXPECT_THROW(partial(output, nullptr), ormi::OperationError);
        EXPECT_THROW(partial(output, input), ormi::ShapeError);
        EXPECT_THROW(partial(output, input, 0), ormi::ShapeError);
        EXPECT_THROW(partial(output, input, 3, 0), ormi::ShapeError);
        EXPECT_THROW(partial(output, input, 0, -3), ormi::ShapeError);
        EXPECT_THROW(partial(output, input, 0, 0, 0), ormi::OperationError);
        EXPECT_THROW(partial(ops::full({}, 1.0, Device::cpu()), input), ormi::ShapeError);
        EXPECT_THROW(partial(ops::full({3, 1}, 1.0, Device::cpu()), input, 0), ormi::ShapeError);
        EXPECT_THROW(partial(ops::full({2, 1}, std::int32_t{1}, Device::cpu()), input, 0), ormi::DTypeError);
        EXPECT_FALSE(input->realized());
        EXPECT_FALSE(output->realized());

        const auto extent = static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) + 2;
        const auto large = ops::full({extent, 1}, 1.0, Device::cpu());
        EXPECT_THROW(partial(large, large), ormi::ShapeError);
        EXPECT_FALSE(large->realized());
    }
}
