#include <array>
#include <limits>
#include <memory>
#include <span>

#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/runtime/dispatch/finite.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace {
    std::shared_ptr<ormi::core::Tensor> tensor(ormi::core::DType dtype) {
        return std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2},
            dtype
        );
    }

    TEST(CPUFiniteDispatchTest, scansMultipleFloatingTensorsInOneCall) {
        const auto first = tensor(ormi::core::DType::float32);
        const auto second = tensor(ormi::core::DType::float64);
        auto* first_data = static_cast<float*>(first->data());
        auto* second_data = static_cast<double*>(second->data());
        first_data[0] = 1.0f;
        first_data[1] = -2.0f;
        second_data[0] = 3.0;
        second_data[1] = 4.0;
        const std::array<const ormi::core::Tensor*, 2> inputs{first.get(), second.get()};

        EXPECT_TRUE(ormi::core::runtime::finite::all(inputs));

        second_data[1] = std::numeric_limits<double>::quiet_NaN();
        EXPECT_FALSE(ormi::core::runtime::finite::all(inputs));
    }

    TEST(FiniteDispatchTest, rejectsInvalidInputSpans) {
        const auto integer = tensor(ormi::core::DType::int32);
        const std::array<const ormi::core::Tensor*, 1> null_input{nullptr};
        const std::array<const ormi::core::Tensor*, 1> integer_input{integer.get()};

        EXPECT_THROW(
            (void)ormi::core::runtime::finite::all(
                std::span<const ormi::core::Tensor* const>{}
            ),
            ormi::OperationError
        );
        EXPECT_THROW(
            (void)ormi::core::runtime::finite::all(null_input),
            ormi::OperationError
        );
        EXPECT_THROW(
            (void)ormi::core::runtime::finite::all(integer_input),
            ormi::DTypeError
        );
    }
}
