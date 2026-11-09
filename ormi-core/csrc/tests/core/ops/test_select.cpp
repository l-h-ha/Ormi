#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/autograd.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/index.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/plan.h>

#include "backends/cpu/index.h"
#include "tests/nn/helpers.h"

namespace {
    using namespace ormi::core;

    TEST(SelectTest, aliasesTheLogicalElementWithoutRealizingItsParent) {
        const auto input = nn_test::leaf({2, 3});
        const std::array values{1., 2., 3., 4., 5., 6.};
        std::copy(values.begin(), values.end(), static_cast<double*>(input->data()));
        const auto squared = ops::mul(input, input);
        const auto transposed = ops::transpose(squared, {1, 0});
        const auto selected = ops::select(transposed, 3);
        EXPECT_TRUE(selected->shape().scalar());
        EXPECT_FALSE(squared->realized());
        EXPECT_FALSE(selected->realized());
        EXPECT_EQ(selected->op()->output, Output::alias);

        Plan plan({selected});
        plan.run();
        EXPECT_DOUBLE_EQ(*static_cast<const double*>(selected->data()), 25.);
        EXPECT_EQ(selected->buffer(), squared->buffer());
        EXPECT_EQ(selected->offset_bytes(), 4 * sizeof(double));
        static_cast<double*>(input->data())[4] = 7.;
        plan.run();
        EXPECT_DOUBLE_EQ(*static_cast<const double*>(selected->data()), 49.);
    }

    TEST(SelectTest, scattersGradientsInLogicalOrderAndDifferentiatesAgain) {
        const auto input = nn_test::leaf({2, 3});
        const std::array values{1., 2., 3., 4., 5., 6.};
        std::copy(values.begin(), values.end(), static_cast<double*>(input->data()));
        const auto transposed = ops::transpose(input, {1, 0});
        const auto cube = ops::mul(ops::mul(transposed, transposed), transposed);
        const auto output = ops::select(cube, 3);
        const auto first = grad(output, input);
        const auto second = grad(ops::sum(first), input);
        const auto third = grad(ops::sum(second), input);
        const auto fourth = grad(ops::sum(third), input);
        EXPECT_FALSE(first->realized());
        EXPECT_FALSE(fourth->realized());
        Plan({first, second, third, fourth}).run();
        const std::array derivatives{75., 30., 6., 0.};
        const std::array roots{first, second, third, fourth};
        for (std::size_t order = 0; order < roots.size(); ++order) {
            const auto* data = static_cast<const double*>(roots[order]->data());
            for (std::size_t i = 0; i < input->numel(); ++i) {
                const auto offset = ormi::backends::cpu::storage_offset(i, *roots[order]);
                EXPECT_DOUBLE_EQ(data[offset], i == 4 ? derivatives[order] : 0.);
            }
        }
    }

    TEST(SelectTest, handlesExistingOffsetsAndZeroStrides) {
        const auto base = nn_test::leaf({8});
        auto* values = static_cast<double*>(base->data());
        for (std::size_t i = 0; i < base->numel(); ++i) {
            values[i] = static_cast<double>(i + 1);
        }
        const auto view = std::make_shared<Tensor>(
            base->buffer(),
            Shape{2, 2},
            DType::float64,
            std::vector<std::int64_t>{3, 1},
            sizeof(double)
        );
        const auto selected = ops::select(view, 3);
        const auto nested = ops::select(selected, 0);
        EXPECT_DOUBLE_EQ(*static_cast<const double*>(nested->data()), 6.);
        EXPECT_EQ(nested->offset_bytes(), 5 * sizeof(double));

        const auto scalar = nn_test::leaf({}, 2.);
        const auto expanded = ops::expand(scalar, {2, 3});
        const auto alias = ops::select(expanded, 4);
        EXPECT_DOUBLE_EQ(*static_cast<const double*>(alias->data()), 2.);
        EXPECT_EQ(alias->offset_bytes(), scalar->offset_bytes());
        const auto value = ops::select(ops::mul(expanded, expanded), 4);
        const auto first = grad(value, scalar);
        const auto second = grad(first, scalar);
        Plan({first, second}).run();
        EXPECT_DOUBLE_EQ(*static_cast<const double*>(first->data()), 4.);
        EXPECT_DOUBLE_EQ(*static_cast<const double*>(second->data()), 2.);
    }

    TEST(SelectTest, retainsStorageAndDoesNotChangeOtherSelectedValues) {
        auto input = nn_test::leaf({3}, 2.);
        auto first = ops::select(input, 0);
        const auto last = ops::select(input, 2);
        const auto loss = ops::mul(first, last);
        const auto gradient = grad(loss, input);
        Plan plan({first, last, gradient});
        input.reset();
        first.reset();
        plan.run();
        const auto* values = static_cast<const double*>(gradient->data());
        EXPECT_DOUBLE_EQ(values[0], 2.);
        EXPECT_DOUBLE_EQ(values[1], 0.);
        EXPECT_DOUBLE_EQ(values[2], 2.);
        EXPECT_DOUBLE_EQ(*static_cast<const double*>(last->data()), 2.);
    }

    TEST(SelectTest, acceptsLogicalIndicesBeyondInt32WithoutAllocatingTheExpandedShape) {
        const auto extent = static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) + 4;
        const auto scalar = ops::full({}, 3., Device::cpu());
        const auto expanded = ops::expand(scalar, {extent});
        const auto selected = ops::select(expanded, static_cast<std::size_t>(extent - 1));
        EXPECT_FALSE(scalar->realized());
        EXPECT_FALSE(selected->realized());
        EXPECT_DOUBLE_EQ(*static_cast<const double*>(selected->data()), 3.);
        EXPECT_EQ(selected->buffer()->bytes(), sizeof(double));
    }

    TEST(SelectTest, rejectsInvalidIndicesBeforeExecution) {
        const auto input = ops::full({2, 3}, 1., Device::cpu());
        EXPECT_THROW(ops::select(nullptr, 0), ormi::OperationError);
        EXPECT_THROW(ops::select(input, 6), ormi::ShapeError);
        EXPECT_THROW(ops::select(input, std::numeric_limits<std::size_t>::max()), ormi::ShapeError);
        EXPECT_THROW(ops::select(ops::full({0, 3}, 1., Device::cpu()), 0), ormi::ShapeError);
        EXPECT_FALSE(input->realized());
    }
}
