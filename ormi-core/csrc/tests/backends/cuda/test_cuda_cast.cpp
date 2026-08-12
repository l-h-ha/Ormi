#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <ormi/core/autograd.h>
#include <ormi/core/attrs.h>
#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/memory/buffer.h>
#include <ormi/core/memory/ops.h>
#include <ormi/core/ops/cast.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace {
    template <typename T>
    ormi::core::DType dtype();

    template <>
    ormi::core::DType dtype<float>() {
        return ormi::core::DType::float32;
    }

    template <>
    ormi::core::DType dtype<double>() {
        return ormi::core::DType::float64;
    }

    template <>
    ormi::core::DType dtype<ormi::core::Float16>() {
        return ormi::core::DType::float16;
    }

    template <>
    ormi::core::DType dtype<std::int32_t>() {
        return ormi::core::DType::int32;
    }

    class CUDACastTest : public ::testing::Test {
    protected:
        void SetUp() override {
            int device_count = 0;
            const cudaError_t status = cudaGetDeviceCount(&device_count);

            if (status != cudaSuccess) {
                GTEST_SKIP() << "CUDA runtime is unavailable: " << cudaGetErrorString(status);
            }

            if (device_count == 0) {
                GTEST_SKIP() << "No CUDA device is available.";
            }
        }

        template <typename T>
        static std::shared_ptr<ormi::core::Tensor> tensor(
            ormi::core::Shape shape,
            const std::vector<T>& values
        ) {
            const auto bytes = values.size() * sizeof(T);
            auto device_buffer = std::make_shared<ormi::core::memory::Buffer>(
                ormi::core::Device::from("cuda"),
                bytes
            );
            ormi::core::memory::Buffer host(ormi::core::Device::cpu(), bytes);

            if (bytes != 0) {
                std::memcpy(host.data(), values.data(), bytes);
                ormi::core::memory::copy(*device_buffer, host);
            }

            return std::make_shared<ormi::core::Tensor>(
                std::move(device_buffer),
                std::move(shape),
                dtype<T>()
            );
        }

        template <typename T>
        static std::vector<T> host_values(ormi::core::Tensor& tensor) {
            std::vector<T> values(tensor.numel());
            ormi::core::memory::Buffer host(
                ormi::core::Device::cpu(),
                values.size() * sizeof(T)
            );
            ormi::core::memory::copy(host, *tensor.buffer());

            if (host.bytes() != 0) {
                std::memcpy(values.data(), host.data(), host.bytes());
            }

            return values;
        }

        static double host_scalar(ormi::core::Tensor& tensor) {
            ormi::core::memory::Buffer host(
                ormi::core::Device::cpu(),
                ormi::core::dtype_size(tensor.dtype())
            );
            ormi::core::memory::copy(host, *tensor.buffer());

            switch (tensor.dtype()) {
                case ormi::core::DType::float64:
                    return *static_cast<const double*>(host.data());
                case ormi::core::DType::float32:
                    return *static_cast<const float*>(host.data());
                case ormi::core::DType::float16:
                    return static_cast<float>(
                        *static_cast<const ormi::core::Float16*>(host.data())
                    );
                case ormi::core::DType::int32:
                    return *static_cast<const std::int32_t*>(host.data());
                case ormi::core::DType::boolean:
                    return *static_cast<const bool*>(host.data()) ? 1.0 : 0.0;
            }

            return 0.0;
        }
    };

    TEST_F(CUDACastTest, convertsEveryDTypePair) {
        const auto device = ormi::core::Device::from("cuda");
        const std::array<std::shared_ptr<ormi::core::Tensor>, 5> inputs{
            ormi::core::ops::full({}, ormi::core::Scalar{1.0}, device),
            ormi::core::ops::full({}, ormi::core::Scalar{1.0f}, device),
            ormi::core::ops::full(
                {},
                ormi::core::Scalar{ormi::core::Float16{1.0f}},
                device
            ),
            ormi::core::ops::full({}, ormi::core::Scalar{std::int32_t{1}}, device),
            ormi::core::ops::full({}, ormi::core::Scalar{true}, device)
        };
        constexpr std::array dtypes{
            ormi::core::DType::float64,
            ormi::core::DType::float32,
            ormi::core::DType::float16,
            ormi::core::DType::int32,
            ormi::core::DType::boolean
        };

        for (const auto& input : inputs) {
            for (const auto output_dtype : dtypes) {
                auto output = ormi::core::ops::cast(input, output_dtype);
                EXPECT_DOUBLE_EQ(host_scalar(*output), 1.0);
            }
        }
    }

    TEST_F(CUDACastTest, truncatesFiniteFloatToInt32) {
        auto result = ormi::core::ops::cast(
            tensor<double>({3}, {-2.75, 0.0, 3.9}),
            ormi::core::DType::int32
        );

        EXPECT_EQ(
            host_values<std::int32_t>(*result),
            (std::vector<std::int32_t>{-2, 0, 3})
        );
    }

    TEST_F(CUDACastTest, acceptsExactInt32Boundaries) {
        auto result = ormi::core::ops::cast(
            tensor<double>(
                {2},
                {
                    static_cast<double>(std::numeric_limits<std::int32_t>::min()),
                    static_cast<double>(std::numeric_limits<std::int32_t>::max())
                }
            ),
            ormi::core::DType::int32
        );

        EXPECT_EQ(
            host_values<std::int32_t>(*result),
            (std::vector<std::int32_t>{
                std::numeric_limits<std::int32_t>::min(),
                std::numeric_limits<std::int32_t>::max()
            })
        );
    }

    TEST_F(CUDACastTest, rejectsInvalidInt32ConversionTransactionally) {
        const std::array invalid_values{
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::quiet_NaN(),
            static_cast<double>(std::numeric_limits<std::int32_t>::max()) + 1.0
        };

        for (const auto value : invalid_values) {
            auto result = ormi::core::ops::cast(
                tensor<double>({1}, {value}),
                ormi::core::DType::int32
            );

            EXPECT_FALSE(result->realized());
            EXPECT_THROW(result->realize(), ormi::DTypeError);
            EXPECT_FALSE(result->realized());
        }

        auto valid = ormi::core::ops::cast(
            tensor<float>({1}, {7.5f}),
            ormi::core::DType::int32
        );
        EXPECT_EQ(host_values<std::int32_t>(*valid)[0], 7);
    }

    TEST_F(CUDACastTest, readsTransposedInputByItsStrides) {
        const auto input = tensor<float>(
            {2, 3},
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}
        );
        const auto transposed = ormi::core::ops::transpose(input, {1, 0});
        auto result = ormi::core::ops::cast(transposed, ormi::core::DType::float64);

        EXPECT_EQ(
            host_values<double>(*result),
            (std::vector<double>{1.0, 4.0, 2.0, 5.0, 3.0, 6.0})
        );
    }

    TEST_F(CUDACastTest, enablesMixedDTypePointwisePromotion) {
        const auto half = tensor<ormi::core::Float16>(
            {2},
            {ormi::core::Float16{1.5f}, ormi::core::Float16{2.0f}}
        );
        const auto single = tensor<float>({2}, {0.5f, 3.0f});
        auto result = ormi::core::ops::add(half, single);

        EXPECT_EQ(result->dtype(), ormi::core::DType::float32);
        EXPECT_EQ(host_values<float>(*result), (std::vector<float>{2.0f, 5.0f}));
    }

    TEST_F(CUDACastTest, differentiatesThroughFloat16CastTwice) {
        const auto input = tensor<float>(ormi::core::Shape{}, {3.0f});
        const auto half = ormi::core::ops::cast(input, ormi::core::DType::float16);
        const auto square = ormi::core::ops::mul(half, half);
        auto second = ormi::core::diff(square, input, 2);

        EXPECT_EQ(second->dtype(), ormi::core::DType::float32);
        EXPECT_FALSE(second->realized());
        EXPECT_FLOAT_EQ(host_values<float>(*second)[0], 2.0f);
    }

    TEST_F(CUDACastTest, realizesEmptyOutputWithoutLaunching) {
        const auto input = ormi::core::ops::full(
            {0},
            2.0f,
            ormi::core::Device::from("cuda")
        );
        auto result = ormi::core::ops::cast(input, ormi::core::DType::float16);

        EXPECT_NO_THROW(result->realize());
        EXPECT_TRUE(result->realized());
        EXPECT_EQ(result->data(), nullptr);
    }

    TEST_F(CUDACastTest, rejectsOversizedIndexedRankLazily) {
        std::vector<std::int64_t> dims(65, 1);
        dims[0] = 2;
        const auto input = ormi::core::ops::expand(
            ormi::core::ops::full(
                ormi::core::Shape{},
                2.0f,
                ormi::core::Device::from("cuda")
            ),
            ormi::core::Shape{std::move(dims)}
        );
        auto result = ormi::core::ops::cast(input, ormi::core::DType::float64);

        EXPECT_FALSE(result->realized());
        EXPECT_THROW(result->realize(), ormi::ShapeError);
        EXPECT_FALSE(result->realized());
    }
}
