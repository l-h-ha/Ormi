#include <array>
#include <memory>
#include <utility>

#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "core/runtime/fusion/expr.h"
#include "core/runtime/fusion/key.h"

namespace {
    std::shared_ptr<ormi::core::Tensor> tensor(ormi::core::Shape shape) {
        return std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            std::move(shape),
            ormi::core::DType::float32
        );
    }

    TEST(FusionKeyTest, distinguishesAuxiliaryOutputs) {
        const auto input = tensor({4});
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(input, input),
            input
        );
        const auto sine = ormi::core::ops::sin(mapped);
        const auto cosine = ormi::core::ops::cos(mapped);
        const std::array<ormi::core::Tensor*, 1> auxiliaries{cosine.get()};
        const auto multiple =
            ormi::core::runtime::fusion::build(*sine, auxiliaries);
        const auto single = ormi::core::runtime::fusion::build(*sine);

        ASSERT_TRUE(multiple.has_value());
        ASSERT_TRUE(single.has_value());
        const ormi::core::runtime::fusion::Key key(*multiple);

        EXPECT_FALSE(key.matches(*single));
        EXPECT_NE(
            key.hash(),
            ormi::core::runtime::fusion::Key::hash(*single)
        );
    }

    TEST(FusionKeyTest, matchesEquivalentGraphs) {
        const auto first_input = tensor({2, 3});
        const auto first_output = ormi::core::ops::sin(
            ormi::core::ops::mul(first_input, first_input)
        );
        const auto first =
            ormi::core::runtime::fusion::build(*first_output);

        const auto second_input = tensor({2, 3});
        const auto second_output = ormi::core::ops::sin(
            ormi::core::ops::mul(second_input, second_input)
        );
        const auto second =
            ormi::core::runtime::fusion::build(*second_output);

        ASSERT_TRUE(first.has_value());
        ASSERT_TRUE(second.has_value());
        const ormi::core::runtime::fusion::Key key(*first);

        EXPECT_TRUE(key.matches(*second));
        EXPECT_EQ(
            key.hash(),
            ormi::core::runtime::fusion::Key::hash(*second)
        );
    }

    TEST(FusionKeyTest, includesShape) {
        const auto matrix_input = tensor({2, 3});
        const auto matrix_output = ormi::core::ops::sin(
            ormi::core::ops::mul(matrix_input, matrix_input)
        );
        const auto matrix =
            ormi::core::runtime::fusion::build(*matrix_output);

        const auto vector_input = tensor({6});
        const auto vector_output = ormi::core::ops::sin(
            ormi::core::ops::mul(vector_input, vector_input)
        );
        const auto vector =
            ormi::core::runtime::fusion::build(*vector_output);

        ASSERT_TRUE(matrix.has_value());
        ASSERT_TRUE(vector.has_value());
        ASSERT_EQ(matrix->nodes, vector->nodes);
        const ormi::core::runtime::fusion::Key key(*matrix);

        EXPECT_FALSE(key.matches(*vector));
        EXPECT_NE(
            key.hash(),
            ormi::core::runtime::fusion::Key::hash(*vector)
        );
    }
}
