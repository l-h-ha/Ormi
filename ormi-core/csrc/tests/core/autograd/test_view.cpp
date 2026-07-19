#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/autograd.h>
#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace {
    TEST(AutogradViewTest, differentiatesExpandAndTranspose) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 1},
            ormi::core::DType::float32
        );
        const auto expanded = ormi::core::ops::expand(input, ormi::core::Shape{2, 3});
        const auto transposed = ormi::core::ops::transpose(
            expanded,
            std::vector<std::int64_t>{1, 0}
        );
        const auto output = ormi::core::ops::sum(transposed);
        const auto gradient = ormi::core::grad(output, input);
        const auto* data = static_cast<const float*>(gradient->data());

        EXPECT_FLOAT_EQ(data[0], 3.0f);
        EXPECT_FLOAT_EQ(data[1], 3.0f);
    }
}
