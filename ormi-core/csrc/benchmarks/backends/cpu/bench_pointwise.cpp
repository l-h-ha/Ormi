#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/float16.h>
#include <ormi/core/ops/kind.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/pointwise.h"

namespace {
    enum class Broadcast { scalar, row, column, strided };
    enum class UnaryLayout { dense, strided, row_gap, transposed, expanded, singleton };

    template <typename T>
    void bmUnary(benchmark::State& state, UnaryLayout layout, ormi::core::ops::UnaryOp op) {
        using namespace ormi::core;
        using Value = std::conditional_t<std::is_same_v<T, Float16>, float, T>;
        const auto rows = state.range(0), columns = state.range(1);
        const auto dtype = std::is_same_v<T, float> ? DType::float32 :
            std::is_same_v<T, double> ? DType::float64 :
            std::is_same_v<T, Float16> ? DType::float16 : DType::int32;
        Shape shape{rows, columns};
        std::vector<std::int64_t> strides{columns, 1};
        std::int64_t stored = rows * columns;
        switch (layout) {
            case UnaryLayout::dense:
                break;
            case UnaryLayout::strided:
                strides = {columns * 2, 2};
                stored *= 2;
                break;
            case UnaryLayout::row_gap:
                strides = {columns + 3, 1};
                stored = rows * (columns + 3);
                break;
            case UnaryLayout::transposed:
                strides = {1, rows};
                break;
            case UnaryLayout::expanded:
                strides = {1, 0};
                stored = rows;
                break;
            case UnaryLayout::singleton:
                shape = {rows, 1, columns, 1};
                strides = {columns * 2, 9999999, 2, 8888888};
                stored *= 2;
                break;
        }
        Tensor storage(Device::cpu(), {stored}, dtype);
        auto* data = static_cast<T*>(storage.data());
        for (std::size_t i = 0; i < storage.numel(); ++i) {
            if constexpr (std::is_same_v<T, std::int32_t>) {
                data[i] = static_cast<T>(i % 13 + 1);
            } else {
                data[i] = static_cast<T>(0.125f + static_cast<float>(i % 13) / 32.0f);
            }
        }
        Tensor input(storage.buffer(), shape, dtype, strides);
        Tensor output(Device::cpu(), shape, dtype);
        ormi::backends::cpu::pointwise_unary(op, output, input);
        const auto* actual = static_cast<const T*>(output.data());
        for (std::size_t i = 0; i < input.numel(); ++i) {
            std::size_t offset = 0, linear = i;
            for (std::size_t axis = shape.rank(); axis > 0; --axis) {
                const auto extent = static_cast<std::size_t>(shape[axis - 1]);
                offset += (linear % extent) * static_cast<std::size_t>(strides[axis - 1]);
                linear /= extent;
            }
            const auto value = static_cast<Value>(data[offset]);
            Value reference{};
            switch (op) {
                case ops::UnaryOp::neg: reference = -value; break;
                case ops::UnaryOp::exp: reference = static_cast<Value>(std::exp(value)); break;
                case ops::UnaryOp::log: reference = static_cast<Value>(std::log(value)); break;
                case ops::UnaryOp::sin: reference = static_cast<Value>(std::sin(value)); break;
                case ops::UnaryOp::cos: reference = static_cast<Value>(std::cos(value)); break;
                case ops::UnaryOp::tanh: reference = static_cast<Value>(std::tanh(value)); break;
                case ops::UnaryOp::rsqrt: reference = static_cast<Value>(Value(1) / std::sqrt(value)); break;
                default:
                    state.SkipWithError("Unary reference does not cover this operation.");
                    return;
            }
            const auto expected = static_cast<T>(reference);
            if (std::abs(static_cast<double>(static_cast<Value>(actual[i])) -
                static_cast<double>(static_cast<Value>(expected))) > 1e-6) {
                state.SkipWithError("Unary output differs from reference.");
                return;
            }
        }
        for (auto _ : state) {
            ormi::backends::cpu::pointwise_unary(op, output, input);
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(input.numel()));
    }

    void bmBroadcast(benchmark::State& state, Broadcast layout, bool swap) {
        using namespace ormi::core;
        const auto rows = state.range(0);
        const auto columns = state.range(1);
        const Shape shape{rows, columns};
        const auto stride = layout == Broadcast::strided ? 2 : 1;
        auto storage = std::make_shared<Tensor>(Device::cpu(), Shape{rows, columns * stride}, DType::float32);
        auto lhs = std::make_shared<Tensor>(
            storage->buffer(),
            shape,
            DType::float32,
            std::vector<std::int64_t>{columns * stride, stride}
        );
        Shape rhs_shape;
        if (layout == Broadcast::row || layout == Broadcast::strided) {
            rhs_shape = {columns};
        } else if (layout == Broadcast::column) {
            rhs_shape = {rows, 1};
        }
        auto rhs = std::make_shared<Tensor>(Device::cpu(), rhs_shape, DType::float32);
        Tensor output(Device::cpu(), shape, DType::float32);
        auto* left = static_cast<float*>(storage->data());
        auto* right = static_cast<float*>(rhs->data());
        for (std::size_t i = 0; i < storage->numel(); ++i) {
            left[i] = static_cast<float>(i % 13) * 0.125f;
        }
        for (std::size_t i = 0; i < rhs->numel(); ++i) {
            right[i] = static_cast<float>(i % 7) * 0.25f;
        }
        if (swap) {
            std::swap(lhs, rhs);
        }
        ormi::backends::cpu::pointwise_binary(ops::BinaryOp::sub, output, *lhs, *rhs);
        const auto* actual = static_cast<const float*>(output.data());
        for (std::size_t i = 0; i < output.numel(); ++i) {
            std::size_t index = 0;
            if (layout == Broadcast::row || layout == Broadcast::strided) {
                index = i % static_cast<std::size_t>(columns);
            } else if (layout == Broadcast::column) {
                index = i / static_cast<std::size_t>(columns);
            }
            const auto expected = swap ? right[index] - left[i * stride] : left[i * stride] - right[index];
            if (actual[i] != expected) {
                state.SkipWithError("Broadcast subtraction differs from reference.");
                return;
            }
        }
        for (auto _ : state) {
            ormi::backends::cpu::pointwise_binary(ops::BinaryOp::sub, output, *lhs, *rhs);
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(output.numel()));
    }

    void bmBinary(benchmark::State& state, ormi::core::ops::BinaryOp op) {
        const auto elements = state.range(0);
        ormi::core::Tensor lhs(ormi::core::Device::cpu(), {elements}, ormi::core::DType::float32);
        ormi::core::Tensor rhs(ormi::core::Device::cpu(), {elements}, ormi::core::DType::float32);
        ormi::core::Tensor output(
            ormi::core::Device::cpu(),
            {elements},
            ormi::core::DType::float32
        );
        std::fill_n(
            static_cast<float*>(lhs.data()),
            lhs.numel(),
            0.25f
        );
        std::fill_n(
            static_cast<float*>(rhs.data()),
            rhs.numel(),
            0.5f
        );

        ormi::backends::cpu::pointwise_binary(op, output, lhs, rhs);
        for (auto _ : state) {
            ormi::backends::cpu::pointwise_binary(op, output, lhs, rhs);
            benchmark::ClobberMemory();
        }

        state.SetItemsProcessed(state.iterations() * elements);
    }
}

#define ORMI_CPU_POINTWISE_SIZES(BENCHMARK_NAME) \
    BENCHMARK_NAME->Arg(1)->Arg(1024)->Arg(32 * 1024)

ORMI_CPU_POINTWISE_SIZES(
    BENCHMARK_CAPTURE(bmBinary, add, ormi::core::ops::BinaryOp::add)
        ->Name("backend/cpu/pointwise/add")
);
ORMI_CPU_POINTWISE_SIZES(
    BENCHMARK_CAPTURE(bmBinary, sub, ormi::core::ops::BinaryOp::sub)
        ->Name("backend/cpu/pointwise/sub")
);
ORMI_CPU_POINTWISE_SIZES(
    BENCHMARK_CAPTURE(bmBinary, mul, ormi::core::ops::BinaryOp::mul)
        ->Name("backend/cpu/pointwise/mul")
);
ORMI_CPU_POINTWISE_SIZES(
    BENCHMARK_CAPTURE(bmBinary, div, ormi::core::ops::BinaryOp::div)
        ->Name("backend/cpu/pointwise/div")
);

#undef ORMI_CPU_POINTWISE_SIZES

#define ORMI_CPU_BROADCAST(NAME, LAYOUT, SWAP) \
    BENCHMARK_CAPTURE(bmBroadcast, NAME, Broadcast::LAYOUT, SWAP) \
        ->Name("backend/cpu/pointwise/broadcast/" #NAME) \
        ->Args({1, 31})->Args({32, 31})->Args({32768, 32})->UseRealTime()

ORMI_CPU_BROADCAST(scalar_rhs, scalar, false);
ORMI_CPU_BROADCAST(scalar_lhs, scalar, true);
ORMI_CPU_BROADCAST(row_rhs, row, false);
ORMI_CPU_BROADCAST(row_lhs, row, true);
ORMI_CPU_BROADCAST(column_rhs, column, false);
ORMI_CPU_BROADCAST(column_lhs, column, true);
ORMI_CPU_BROADCAST(strided_rhs, strided, false);
ORMI_CPU_BROADCAST(strided_lhs, strided, true);

#undef ORMI_CPU_BROADCAST

#define ORMI_CPU_UNARY_CASE(TYPE, NAME, OP, LAYOUT) \
    BENCHMARK_CAPTURE(bmUnary<TYPE>, NAME##_##OP##_##LAYOUT, UnaryLayout::LAYOUT, \
        ormi::core::ops::UnaryOp::OP) \
        ->Name("backend/cpu/pointwise/unary/" #NAME "/" #OP "/" #LAYOUT) \
        ->Args({1, 31})->Args({32, 32})->Args({512, 32})->Args({32768, 32})->UseRealTime()

#define ORMI_CPU_UNARY_LAYOUTS(TYPE, NAME, OP) \
    ORMI_CPU_UNARY_CASE(TYPE, NAME, OP, dense); \
    ORMI_CPU_UNARY_CASE(TYPE, NAME, OP, strided); \
    ORMI_CPU_UNARY_CASE(TYPE, NAME, OP, row_gap); \
    ORMI_CPU_UNARY_CASE(TYPE, NAME, OP, transposed); \
    ORMI_CPU_UNARY_CASE(TYPE, NAME, OP, expanded); \
    ORMI_CPU_UNARY_CASE(TYPE, NAME, OP, singleton)

ORMI_CPU_UNARY_LAYOUTS(float, f32, neg);
ORMI_CPU_UNARY_LAYOUTS(float, f32, exp);
ORMI_CPU_UNARY_LAYOUTS(float, f32, log);
ORMI_CPU_UNARY_LAYOUTS(float, f32, sin);
ORMI_CPU_UNARY_LAYOUTS(float, f32, cos);
ORMI_CPU_UNARY_LAYOUTS(float, f32, tanh);
ORMI_CPU_UNARY_LAYOUTS(float, f32, rsqrt);
ORMI_CPU_UNARY_LAYOUTS(double, f64, neg);
ORMI_CPU_UNARY_LAYOUTS(double, f64, exp);
ORMI_CPU_UNARY_LAYOUTS(double, f64, log);
ORMI_CPU_UNARY_LAYOUTS(double, f64, sin);
ORMI_CPU_UNARY_LAYOUTS(double, f64, cos);
ORMI_CPU_UNARY_LAYOUTS(double, f64, tanh);
ORMI_CPU_UNARY_LAYOUTS(double, f64, rsqrt);
ORMI_CPU_UNARY_LAYOUTS(ormi::core::Float16, f16, neg);
ORMI_CPU_UNARY_LAYOUTS(std::int32_t, i32, neg);

#undef ORMI_CPU_UNARY_LAYOUTS
#undef ORMI_CPU_UNARY_CASE

#define ORMI_CPU_UNARY_ONE(TYPE, NAME, OP) \
    BENCHMARK_CAPTURE(bmUnary<TYPE>, NAME##_##OP##_one, UnaryLayout::dense, ormi::core::ops::UnaryOp::OP) \
        ->Name("backend/cpu/pointwise/unary/" #NAME "/" #OP "/dense_one") \
        ->Args({1, 1})->UseRealTime()

ORMI_CPU_UNARY_ONE(float, f32, neg);
ORMI_CPU_UNARY_ONE(float, f32, exp);
ORMI_CPU_UNARY_ONE(double, f64, neg);
ORMI_CPU_UNARY_ONE(ormi::core::Float16, f16, neg);
ORMI_CPU_UNARY_ONE(std::int32_t, i32, neg);

#undef ORMI_CPU_UNARY_ONE
