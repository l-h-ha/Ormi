#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/autograd.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/plan.h>
#include <ormi/core/tensor.h>

namespace {
    using namespace ormi::core;

    TEST(TensorCopyTest, feedsTheSamePlanWithoutChangingStorageOrGraphIdentity) {
        const auto input = std::make_shared<Tensor>(Device::cpu(), Shape{2, 1, 2}, DType::float64);
        const auto output = ops::mul(ops::mul(input, input), input);
        const auto first = grad(ops::sum(output), input);
        const auto second = grad(ops::sum(first), input);
        Plan plan({output, first, second});
        const auto buffer = input->buffer();
        const auto steps = plan.steps();
        const auto storage = plan.storage_bytes();
        for (const auto offset : {0., 2., -1.}) {
            const std::array values{1. + offset, 2. + offset, 3. + offset, 4. + offset};
            input->copy_from(std::as_bytes(std::span(values)));
            EXPECT_EQ(input->buffer(), buffer);
            EXPECT_EQ(input->op(), nullptr);
            EXPECT_TRUE(input->parents().empty());
            plan.run();
            for (std::size_t i = 0; i < values.size(); ++i) {
                EXPECT_DOUBLE_EQ(
                    static_cast<const double*>(output->data())[i],
                    values[i] * values[i] * values[i]
                );
                EXPECT_DOUBLE_EQ(static_cast<const double*>(first->data())[i], 3 * values[i] * values[i]);
                EXPECT_DOUBLE_EQ(static_cast<const double*>(second->data())[i], 6 * values[i]);
            }
            EXPECT_EQ(plan.steps(), steps);
            EXPECT_EQ(plan.storage_bytes(), storage);
        }
    }

    TEST(TensorCopyTest, rejectsIncompatibleOrLazySourcesBeforeWriting) {
        Tensor input(Device::cpu(), {2, 3}, DType::float64);
        const std::array values{1., 2., 3., 4., 5., 6.};
        input.copy_from(std::as_bytes(std::span(values)));
        const auto source = std::make_shared<Tensor>(Device::cpu(), Shape{2, 3}, DType::float64);
        source->copy_from(std::as_bytes(std::span(values)));
        const auto lazy = ops::sin(source);
        EXPECT_THROW(input.copy_from(*lazy), ormi::OperationError);
        EXPECT_FALSE(lazy->realized());
        EXPECT_THROW(input.copy_from(Tensor(Device::cpu(), {3, 2}, DType::float64)), ormi::ShapeError);
        EXPECT_THROW(input.copy_from(Tensor(Device::cpu(), {2, 3}, DType::float32)), ormi::DTypeError);
        EXPECT_THROW(input.copy_from(std::as_bytes(std::span(values).first(5))), ormi::ShapeError);
        auto strided = ops::transpose(source, {1, 0});
        strided->realize();
        Tensor transposed_shape(Device::cpu(), {3, 2}, DType::float64);
        EXPECT_THROW(transposed_shape.copy_from(*strided), ormi::OperationError);
        EXPECT_THROW(lazy->copy_from(input), ormi::OperationError);
        lazy->realize();
        EXPECT_THROW(lazy->copy_from(input), ormi::OperationError);
        for (std::size_t i = 0; i < values.size(); ++i) {
            EXPECT_DOUBLE_EQ(static_cast<const double*>(input.data())[i], values[i]);
        }
    }

    TEST(TensorCopyTest, handlesEmptyTensorsAndSelfCopies) {
        Tensor empty(Device::cpu(), {2, 0, 3}, DType::int32);
        Tensor other(Device::cpu(), {2, 0, 3}, DType::int32);
        EXPECT_NO_THROW(empty.copy_from(other));
        EXPECT_NO_THROW(empty.copy_from(std::span<const std::byte>{}));
        Tensor scalar(Device::cpu(), {}, DType::float64);
        const std::array value{7.};
        scalar.copy_from(std::as_bytes(std::span(value)));
        scalar.copy_from(scalar);
        scalar.copy_from(std::as_bytes(std::span(static_cast<const double*>(scalar.data()), 1)));
        EXPECT_DOUBLE_EQ(*static_cast<const double*>(scalar.data()), 7.);
    }

    TEST(TensorCopyTest, rejectsOverlappingRangesButAllowsDisjointViewsOfOneBuffer) {
        const auto owner = std::make_shared<Tensor>(Device::cpu(), Shape{8}, DType::float64);
        const std::array values{1., 2., 3., 4., 5., 6., 7., 8.};
        owner->copy_from(std::as_bytes(std::span(values)));
        Tensor destination(owner->buffer(), {4}, DType::float64);
        Tensor overlap(owner->buffer(), {4}, DType::float64, {}, sizeof(double));
        EXPECT_THROW(destination.copy_from(overlap), ormi::OperationError);
        const auto* pointer = static_cast<const double*>(owner->data());
        EXPECT_THROW(destination.copy_from(std::as_bytes(std::span(pointer + 1, 4))), ormi::OperationError);
        Tensor disjoint(owner->buffer(), {4}, DType::float64, {}, 4 * sizeof(double));
        destination.copy_from(disjoint);
        for (std::size_t i = 0; i < 4; ++i) {
            EXPECT_DOUBLE_EQ(static_cast<const double*>(destination.data())[i], values[i + 4]);
        }
        Tensor strided(owner->buffer(), {2, 2}, DType::float64, {3, 1});
        EXPECT_THROW(strided.copy_from(std::as_bytes(std::span(values).first(4))), ormi::OperationError);
    }
}
