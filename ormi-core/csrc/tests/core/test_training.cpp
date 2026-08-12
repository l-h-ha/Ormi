#include <cstddef>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/autograd.h>
#include <ormi/core/device.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/linalg.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/tensor.h>

namespace {
    TEST(CPUTrainingTest, realizesTwoLayerGradients) {
        const auto device = ormi::core::Device::cpu();
        const auto input =
            ormi::core::ops::full({1024, 16}, 0.25f, device);
        const auto target =
            ormi::core::ops::full({1024, 1}, 0.5f, device);
        const std::vector<std::shared_ptr<ormi::core::Tensor>> parameters{
            ormi::core::ops::full({16, 32}, 0.125f, device),
            ormi::core::ops::full({32}, 0.01f, device),
            ormi::core::ops::full({32, 1}, 0.125f, device),
            ormi::core::ops::full({1}, 0.01f, device)
        };
        const auto hidden = ormi::core::ops::sin(
            ormi::core::ops::add(
                ormi::core::ops::matmul(input, parameters[0]),
                parameters[1]
            )
        );
        const auto prediction = ormi::core::ops::add(
            ormi::core::ops::matmul(hidden, parameters[2]),
            parameters[3]
        );
        const auto error =
            ormi::core::ops::sub(prediction, target);
        const auto loss = ormi::core::ops::mean(
            ormi::core::ops::mul(error, error)
        );
        const auto gradients =
            ormi::core::grad(loss, parameters);

        loss->realize();
        for (const auto& gradient : gradients) {
            gradient->realize();
        }

        ASSERT_EQ(gradients.size(), parameters.size());
        for (std::size_t i = 0; i < gradients.size(); ++i) {
            EXPECT_EQ(gradients[i]->shape(), parameters[i]->shape());
            const auto finite =
                ormi::core::ops::all_finite(gradients[i]);
            EXPECT_TRUE(*static_cast<const bool*>(finite->data()));
        }
    }
}
