#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <limits>

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
    TEST(IndexTest, slicesViewsReversedCopiesAndSelectedAxes) {
        using namespace ormi::core;
        const auto input = nn_test::leaf({2, 4});
        auto* values = static_cast<double*>(input->data());

        for (int i = 0; i < 8; ++i) {
            values[i] = static_cast<double>(i);
        }

        const auto selected = ops::slice(input, 1, 1, 2, 2);
        const auto reversed = ops::slice(input, 1, 3, 4, -1);
        const auto row = ops::select_axis(input, 0, -1);

        EXPECT_FALSE(selected->realized());
        EXPECT_FALSE(reversed->realized());
        EXPECT_EQ(selected->shape(), (Shape{2, 2}));
        EXPECT_EQ(row->shape(), (Shape{4}));
        selected->realize();

        EXPECT_EQ(selected->buffer(), input->buffer());
        EXPECT_EQ(*static_cast<const double*>(selected->data()), 1.);
        reversed->realize();
        const std::array expected{3., 2., 1., 0., 7., 6., 5., 4.};

        EXPECT_TRUE(std::equal(expected.begin(), expected.end(), static_cast<const double*>(reversed->data())));
        EXPECT_EQ(*static_cast<const double*>(row->data()), 4.);
    }

    TEST(IndexTest, concatenatesManyStridedInputsAndEmptySlices) {
        using namespace ormi::core;
        const auto input = nn_test::leaf({2, 3});
        auto* values = static_cast<double*>(input->data());

        for (int i = 0; i < 6; ++i) {
            values[i] = static_cast<double>(i);
        }

        const auto left = ops::slice(input, 1, 0, 2, 2);
        const auto empty = ops::slice(input, 1, 0, 0);
        const auto output = ops::concat({left, empty, input}, -1);

        EXPECT_FALSE(output->realized());
        EXPECT_EQ(output->shape(), (Shape{2, 5}));
        const std::array expected{0., 2., 0., 1., 2., 3., 5., 3., 4., 5.};

        EXPECT_TRUE(std::equal(expected.begin(), expected.end(), static_cast<const double*>(output->data())));
    }

    TEST(IndexTest, sliceAndConcatPreserveThirdDerivativesAndReplay) {
        using namespace ormi::core;
        const auto input = nn_test::leaf({2, 3}, 0.5);
        const auto reversed = ops::slice(input, 1, 2, 3, -1);
        const auto joined = ops::concat({input, reversed}, 0);
        const auto loss = ops::sum(ops::mul(ops::mul(joined, joined), joined));
        const auto first = grad(loss, input);
        const auto second = grad(ops::sum(first), input);
        const auto third = grad(ops::sum(second), input);
        Plan plan({loss, first, second, third});

        EXPECT_FALSE(loss->realized());

        for (const double value : {0.5, -0.25}) {
            std::fill_n(static_cast<double*>(input->data()), 6, value);
            plan.run();

            for (int i = 0; i < 6; ++i) {
                EXPECT_DOUBLE_EQ(static_cast<const double*>(first->data())[i], 6.*value*value);
                EXPECT_DOUBLE_EQ(static_cast<const double*>(second->data())[i], 12.*value);
                EXPECT_DOUBLE_EQ(static_cast<const double*>(third->data())[i], 12.);
            }
        }
    }

    TEST(IndexTest, validatesSliceAndConcatWithoutAllocatingHugeShapes) {
        using namespace ormi::core;
        const auto input = nn_test::leaf({2, 3});

        EXPECT_THROW(ops::slice(nullptr, 0, 0, 1), ormi::OperationError);
        EXPECT_THROW(ops::slice(input, 2, 0, 1), ormi::ShapeError);
        EXPECT_THROW(ops::slice(input, 0, 0, -1), ormi::ShapeError);
        EXPECT_THROW(ops::slice(input, 0, 0, 1, 0), ormi::ShapeError);
        EXPECT_THROW(ops::slice(input, 0, 1, 2), ormi::ShapeError);
        EXPECT_THROW(ops::slice(input, 0, 0, 2, -1), ormi::ShapeError);
        EXPECT_THROW(ops::concat({}), ormi::OperationError);
        EXPECT_THROW(ops::concat({input, nullptr}), ormi::OperationError);
        EXPECT_THROW(ops::concat({input, nn_test::leaf({3, 2})}), ormi::ShapeError);
        const auto empty = std::make_shared<Tensor>(Device::cpu(), Shape{0, std::numeric_limits<std::int64_t>::max()}, DType::float64);

        EXPECT_THROW(ops::concat({empty, empty}, 1), ormi::ShapeError);
    }

    TEST(IndexTest, scattersLazilyAndRetainsHigherOrderAdjoints) {
        using namespace ormi::core;
        const auto input = nn_test::leaf({}, 3.0);
        const auto indices = ops::full({}, std::int32_t{1}, Device::cpu());
        const auto output = ops::unpick(input, indices, {3});
        const auto first = grad(ops::sum(ops::mul(output, output)), input);
        const auto second = grad(first, input);

        EXPECT_FALSE(output->realized());
        EXPECT_FALSE(first->realized());
        Plan({output, first, second}).run();
        const auto* values = static_cast<const double*>(output->data());

        EXPECT_DOUBLE_EQ(values[0], 0.0);
        EXPECT_DOUBLE_EQ(values[1], 3.0);
        EXPECT_DOUBLE_EQ(values[2], 0.0);
        EXPECT_DOUBLE_EQ(*static_cast<const double*>(first->data()), 6.0);
        EXPECT_DOUBLE_EQ(*static_cast<const double*>(second->data()), 2.0);
        EXPECT_THROW(ops::unpick(input, indices, {}), ormi::ShapeError);
        EXPECT_THROW(ops::unpick(input, indices, {2, 3}), ormi::ShapeError);
        EXPECT_THROW(ops::unpick(nullptr, indices, {3}), ormi::OperationError);
    }

    TEST(IndexTest, selectsStridedSlicesAndDifferentiatesTwice) {
        const auto base = nn_test::leaf({3, 2});
        const std::array values{1., 2., 3., 4., 5., 6.};
        std::copy(values.begin(), values.end(), static_cast<double*>(base->data()));
        const auto input = ormi::core::ops::transpose(base, {1, 0});
        const auto indices = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2},
            ormi::core::DType::int32
        );
        auto* labels = static_cast<std::int32_t*>(indices->data());
        labels[0] = 2;
        labels[1] = 0;
        const auto selected = ormi::core::ops::pick(input, indices);
        const auto loss = ormi::core::ops::sum(ormi::core::ops::mul(selected, selected));
        const auto gradient = ormi::core::grad(loss, base);
        const auto second = ormi::core::grad(ormi::core::ops::sum(gradient), base);
        ormi::core::Plan({selected, gradient, second}).run();

        EXPECT_DOUBLE_EQ(static_cast<const double*>(selected->data())[0], 5.);
        EXPECT_DOUBLE_EQ(static_cast<const double*>(selected->data())[1], 2.);
        const std::array expected{0., 4., 0., 0., 10., 0.};

        for (std::size_t i = 0; i < expected.size(); ++i) {
            const auto first_offset = ormi::backends::cpu::storage_offset(i, *gradient);
            const auto second_offset = ormi::backends::cpu::storage_offset(i, *second);

            EXPECT_DOUBLE_EQ(static_cast<const double*>(gradient->data())[first_offset], expected[i]);
            EXPECT_DOUBLE_EQ(
                static_cast<const double*>(second->data())[second_offset],
                expected[i] == 0 ? 0 : 2
            );
        }
    }

    TEST(IndexTest, invalidIndicesProduceNaNAndIgnoredIndicesProduceZero) {
        const auto input = nn_test::leaf({2, 3});
        const auto indices = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2},
            ormi::core::DType::int32
        );
        auto* labels = static_cast<std::int32_t*>(indices->data());
        labels[0] = -100;
        labels[1] = 3;
        const auto result = ormi::core::ops::pick(input, indices, -1, -100);
        const auto* values = static_cast<const double*>(result->data());

        EXPECT_DOUBLE_EQ(values[0], 0.);
        EXPECT_TRUE(std::isnan(values[1]));
        EXPECT_THROW(ormi::core::ops::pick(input, indices, 2), ormi::ShapeError);
    }
}
