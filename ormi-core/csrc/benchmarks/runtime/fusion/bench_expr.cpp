#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <benchmark/benchmark.h>

#include <ormi/core/autocast.h>
#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/float16.h>
#include <ormi/core/op.h>
#include <ormi/core/ops/linalg.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/jit/kernel.h"
#include "core/runtime/fusion/expr.h"

namespace {
    enum class InputShape {
        equal,
        scalar,
        row,
        column
    };

    enum class Reduction {
        sum,
        mean
    };

    std::shared_ptr<ormi::core::Tensor> tensor(
        ormi::core::Shape shape,
        ormi::core::DType dtype = ormi::core::DType::float32
    ) {
        auto result = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            std::move(shape),
            dtype
        );

        if (dtype == ormi::core::DType::float16) {
            std::fill_n(
                static_cast<ormi::core::Float16*>(result->data()),
                result->numel(),
                ormi::core::Float16{0.5f}
            );
        } else {
            std::fill_n(static_cast<float*>(result->data()), result->numel(), 0.5f);
        }
        return result;
    }

    ormi::core::Shape rhs_shape(
        InputShape input_shape,
        std::int64_t rows,
        std::int64_t columns
    ) {
        switch (input_shape) {
            case InputShape::equal:
                return ormi::core::Shape{rows, columns};
            case InputShape::scalar:
                return ormi::core::Shape{};
            case InputShape::row:
                return ormi::core::Shape{columns};
            case InputShape::column:
                return ormi::core::Shape{rows, 1};
        }

        throw std::logic_error("Unknown fusion benchmark input shape.");
    }

    std::shared_ptr<ormi::core::Tensor> broadcast_chain(
        const std::shared_ptr<ormi::core::Tensor>& lhs,
        const std::shared_ptr<ormi::core::Tensor>& rhs
    ) {
        return ormi::core::ops::sin(
            ormi::core::ops::add(ormi::core::ops::mul(lhs, lhs), rhs)
        );
    }

    std::shared_ptr<ormi::core::Tensor> broadcast_arithmetic_chain(
        const std::shared_ptr<ormi::core::Tensor>& lhs,
        const std::shared_ptr<ormi::core::Tensor>& rhs
    ) {
        const auto square = ormi::core::ops::mul(lhs, lhs);
        const auto added = ormi::core::ops::add(square, rhs);
        return ormi::core::ops::mul(added, lhs);
    }

    std::shared_ptr<ormi::core::Tensor> arithmetic_chain(
        const std::shared_ptr<ormi::core::Tensor>& input
    ) {
        const auto square = ormi::core::ops::mul(input, input);
        const auto added = ormi::core::ops::add(square, input);
        return ormi::core::ops::mul(added, input);
    }

    std::shared_ptr<ormi::core::Tensor> scalar_arithmetic_chain(
        const std::shared_ptr<ormi::core::Tensor>& input,
        const std::shared_ptr<ormi::core::Tensor>& scale,
        const std::shared_ptr<ormi::core::Tensor>& bias
    ) {
        const auto scaled = ormi::core::ops::mul(input, scale);
        const auto shifted = ormi::core::ops::add(scaled, bias);
        return ormi::core::ops::mul(shifted, input);
    }

    void bmBroadcastRealize(benchmark::State& state, InputShape input_shape) {
        const auto rows = state.range(0);
        const auto columns = state.range(1);
        const auto lhs = tensor({rows, columns});
        const auto rhs = tensor(rhs_shape(input_shape, rows, columns));

        auto warmup = broadcast_chain(lhs, rhs);
        warmup->realize();

        for (auto _ : state) {
            state.PauseTiming();
            auto output = broadcast_chain(lhs, rhs);
            state.ResumeTiming();

            output->realize();
            benchmark::DoNotOptimize(output->data());
        }

        state.SetItemsProcessed(state.iterations() * rows * columns);
    }

    void bmUnfusedBroadcastRealize(benchmark::State& state, InputShape input_shape) {
        const auto rows = state.range(0);
        const auto columns = state.range(1);
        const auto lhs = tensor({rows, columns});
        const auto rhs = tensor(rhs_shape(input_shape, rows, columns));

        for (auto _ : state) {
            state.PauseTiming();
            auto square = ormi::core::ops::mul(lhs, lhs);
            auto sum = ormi::core::ops::add(square, rhs);
            auto output = ormi::core::ops::sin(sum);
            state.ResumeTiming();

            square->realize();
            sum->realize();
            output->realize();
            benchmark::DoNotOptimize(output->data());
        }

        state.SetItemsProcessed(state.iterations() * rows * columns);
    }

    void bmBroadcastArithmeticRealize(
        benchmark::State& state,
        InputShape input_shape,
        ormi::core::DType dtype = ormi::core::DType::float32
    ) {
        const auto rows = state.range(0);
        const auto columns = state.range(1);
        const auto lhs = tensor({rows, columns}, dtype);
        const auto rhs = tensor(rhs_shape(input_shape, rows, columns), dtype);

        auto warmup = broadcast_arithmetic_chain(lhs, rhs);
        warmup->realize();

        for (auto _ : state) {
            state.PauseTiming();
            auto output = broadcast_arithmetic_chain(lhs, rhs);
            state.ResumeTiming();

            output->realize();
            benchmark::DoNotOptimize(output->data());
        }

        state.SetItemsProcessed(state.iterations() * rows * columns);
    }

    void bmTransposeRealize(benchmark::State& state, bool fused) {
        const auto rows = state.range(0);
        const auto columns = state.range(1);
        const auto base = tensor({columns, rows});
        const auto input = ormi::core::ops::transpose(base, {1, 0});

        if (fused) {
            auto warmup = broadcast_chain(input, input);
            warmup->realize();
        }

        for (auto _ : state) {
            state.PauseTiming();
            auto square = ormi::core::ops::mul(input, input);
            auto sum = ormi::core::ops::add(square, input);
            auto output = ormi::core::ops::sin(sum);
            state.ResumeTiming();

            if (!fused) {
                square->realize();
                sum->realize();
            }
            output->realize();
            benchmark::DoNotOptimize(output->data());
        }

        state.SetItemsProcessed(state.iterations() * rows * columns);
    }

    void bmFloat16Realize(benchmark::State& state, bool force_unfused) {
        const auto elements = state.range(0);
        const auto input = tensor({elements}, ormi::core::DType::float16);

        if (!force_unfused) {
            auto warmup = broadcast_chain(input, input);
            warmup->realize();
        }

        for (auto _ : state) {
            state.PauseTiming();
            auto square = ormi::core::ops::mul(input, input);
            auto sum = ormi::core::ops::add(square, input);
            auto output = ormi::core::ops::sin(sum);
            state.ResumeTiming();

            if (force_unfused) {
                square->realize();
                sum->realize();
            }
            output->realize();
            benchmark::DoNotOptimize(output->data());
        }

        state.SetItemsProcessed(state.iterations() * elements);
    }

    void bmFloat16ArithmeticRealize(benchmark::State& state) {
        const auto elements = state.range(0);
        const auto input = tensor({elements}, ormi::core::DType::float16);

        auto warmup = arithmetic_chain(input);
        warmup->realize();

        for (auto _ : state) {
            state.PauseTiming();
            auto output = arithmetic_chain(input);
            state.ResumeTiming();

            output->realize();
            benchmark::DoNotOptimize(output->data());
        }

        state.SetItemsProcessed(state.iterations() * elements);
    }

    void bmArithmeticRealize(benchmark::State& state, bool force_unfused) {
        const auto elements = state.range(0);
        const auto input = tensor({elements});

        if (!force_unfused) {
            auto warmup = arithmetic_chain(input);
            warmup->realize();
        }

        for (auto _ : state) {
            state.PauseTiming();
            auto square = ormi::core::ops::mul(input, input);
            auto added = ormi::core::ops::add(square, input);
            auto output = ormi::core::ops::mul(added, input);
            state.ResumeTiming();

            if (force_unfused) {
                square->realize();
                added->realize();
            }
            output->realize();
            benchmark::DoNotOptimize(output->data());
        }

        state.SetItemsProcessed(state.iterations() * elements);
    }

    void bmScalarArithmeticRealize(benchmark::State& state) {
        const auto elements = state.range(0);
        const auto input = tensor({elements});
        const auto scale = tensor({});
        const auto bias = tensor({});

        auto warmup = scalar_arithmetic_chain(input, scale, bias);
        warmup->realize();

        for (auto _ : state) {
            state.PauseTiming();
            auto output = scalar_arithmetic_chain(input, scale, bias);
            state.ResumeTiming();

            output->realize();
            benchmark::DoNotOptimize(output->data());
        }

        state.SetItemsProcessed(state.iterations() * elements);
    }

    std::shared_ptr<ormi::core::Tensor> reduce(
        const std::shared_ptr<ormi::core::Tensor>& input,
        Reduction reduction
    ) {
        return reduction == Reduction::sum
            ? ormi::core::ops::sum(input)
            : ormi::core::ops::mean(input);
    }

    void bmMapReductionRealize(
        benchmark::State& state,
        Reduction reduction,
        bool force_materialized
    ) {
        const auto elements = state.range(0);
        const auto input = tensor({elements});

        auto warmup_mapped = broadcast_chain(input, input);
        auto warmup = reduce(warmup_mapped, reduction);
        if (force_materialized) {
            warmup_mapped->realize();
        }
        warmup->realize();

        for (auto _ : state) {
            state.PauseTiming();
            auto square = ormi::core::ops::mul(input, input);
            auto added = ormi::core::ops::add(square, input);
            auto mapped = ormi::core::ops::sin(added);
            auto output = reduce(mapped, reduction);
            state.ResumeTiming();

            if (force_materialized) {
                mapped->realize();
            }
            output->realize();
            benchmark::DoNotOptimize(output->data());
        }

        state.SetItemsProcessed(state.iterations() * elements);
    }

    void bmPartialReductionCall(
        benchmark::State& state,
        bool prepared
    ) {
        const auto elements = state.range(0);
        const auto input = tensor({elements});
        const auto root = ormi::core::ops::sum(
            ormi::core::ops::sin(
                ormi::core::ops::add(
                    ormi::core::ops::mul(input, input),
                    input
                )
            )
        );
        const auto expression =
            ormi::core::runtime::fusion::build(*root).value();
        root->realize();
        auto setup = prepared
            ? ormi::backends::cpu::jit::prepare(*root, expression)
            : ormi::core::Setup{};

        for (auto _ : state) {
            if (prepared) {
                setup.run(*root);
            } else {
                ormi::backends::cpu::jit::execute(*root, expression);
            }
            benchmark::ClobberMemory();
        }

        state.SetItemsProcessed(state.iterations() * elements);
        state.counters["storage_buffers"] =
            static_cast<double>(setup.buffers());
        state.counters["storage_bytes"] =
            static_cast<double>(setup.bytes());
        const auto actual =
            static_cast<double>(*static_cast<const float*>(root->data()));
        const auto expected =
            static_cast<double>(std::sin(0.75f)) *
            static_cast<double>(elements);
        state.counters["relative_error_ppm"] =
            std::abs(actual - expected) / std::abs(expected) * 1.0e6;
    }

    void bmVectorizedReductionCall(
        benchmark::State& state,
        bool prepared
    ) {
        const auto elements = state.range(0);
        const auto input = tensor({elements});
        std::fill_n(
            static_cast<float*>(input->data()),
            input->numel(),
            0.1f
        );
        const auto root = ormi::core::ops::sum(arithmetic_chain(input));
        const auto expression =
            ormi::core::runtime::fusion::build(*root).value();
        root->realize();
        auto setup = prepared
            ? ormi::backends::cpu::jit::prepare(*root, expression)
            : ormi::core::Setup{};

        for (auto _ : state) {
            if (prepared) {
                setup.run(*root);
            } else {
                ormi::backends::cpu::jit::execute(*root, expression);
            }
            benchmark::ClobberMemory();
        }

        state.SetItemsProcessed(state.iterations() * elements);
        const auto actual =
            static_cast<double>(*static_cast<const float*>(root->data()));
        const float value = 0.1f;
        const float mapped = (value * value + value) * value;
        const auto expected =
            static_cast<double>(mapped) * static_cast<double>(elements);
        state.counters["relative_error_ppm"] =
            std::abs(actual - expected) / std::abs(expected) * 1.0e6;
    }

    void bmPackedReductionCall(
        benchmark::State& state,
        bool prepared
    ) {
        const auto elements = state.range(0);
        const auto input = tensor(
            {elements},
            ormi::core::DType::float16
        );
        const ormi::core::Float16 value{0.1f};
        std::fill_n(
            static_cast<ormi::core::Float16*>(input->data()),
            input->numel(),
            value
        );
        std::shared_ptr<ormi::core::Tensor> root;
        ormi::core::Float16 mapped;
        {
            const ormi::core::Autocast autocast;
            root = ormi::core::ops::sum(arithmetic_chain(input));
            const ormi::core::Float16 square{
                static_cast<float>(value) * static_cast<float>(value)
            };
            const ormi::core::Float16 added{
                static_cast<float>(square) + static_cast<float>(value)
            };
            mapped = ormi::core::Float16{
                static_cast<float>(added) * static_cast<float>(value)
            };
        }
        const auto expression =
            ormi::core::runtime::fusion::build(*root).value();
        root->realize();
        auto setup = prepared
            ? ormi::backends::cpu::jit::prepare(*root, expression)
            : ormi::core::Setup{};

        for (auto _ : state) {
            if (prepared) {
                setup.run(*root);
            } else {
                ormi::backends::cpu::jit::execute(*root, expression);
            }
            benchmark::ClobberMemory();
        }

        state.SetItemsProcessed(state.iterations() * elements);
        const auto actual =
            static_cast<double>(*static_cast<const float*>(root->data()));
        const auto expected = static_cast<double>(static_cast<float>(mapped)) *
            static_cast<double>(elements);
        state.counters["relative_error_ppm"] =
            std::abs(actual - expected) / std::abs(expected) * 1.0e6;
    }

    void bmArithmeticMapReductionRealize(
        benchmark::State& state,
        Reduction reduction,
        ormi::core::DType dtype = ormi::core::DType::float32
    ) {
        const auto elements = state.range(0);
        const auto input = tensor({elements}, dtype);

        auto warmup = reduce(arithmetic_chain(input), reduction);
        warmup->realize();

        for (auto _ : state) {
            state.PauseTiming();
            auto output = reduce(arithmetic_chain(input), reduction);
            state.ResumeTiming();

            output->realize();
            benchmark::DoNotOptimize(output->data());
        }

        state.SetItemsProcessed(state.iterations() * elements);
    }

    void bmPartialArithmeticReductionRealize(
        benchmark::State& state,
        Reduction reduction,
        std::int64_t axis,
        ormi::core::DType dtype = ormi::core::DType::float32
    ) {
        const auto rows = state.range(0);
        const auto columns = state.range(1);
        const auto input = tensor({rows, columns}, dtype);

        auto warmup = reduction == Reduction::sum
            ? ormi::core::ops::sum(arithmetic_chain(input), {axis})
            : ormi::core::ops::mean(arithmetic_chain(input), {axis});
        warmup->realize();

        for (auto _ : state) {
            state.PauseTiming();
            auto output = reduction == Reduction::sum
                ? ormi::core::ops::sum(arithmetic_chain(input), {axis})
                : ormi::core::ops::mean(arithmetic_chain(input), {axis});
            state.ResumeTiming();

            output->realize();
            benchmark::DoNotOptimize(output->data());
        }

        state.SetItemsProcessed(state.iterations() * rows * columns);
    }

    void bmSeparatedAxesMeanRealize(benchmark::State& state) {
        const auto first = state.range(0);
        const auto middle = state.range(1);
        const auto last = state.range(2);
        const auto input = tensor({first, middle, last});
        const auto row = tensor({last});
        const auto column = tensor({middle, 1});

        auto build = [&] {
            const auto scaled = ormi::core::ops::mul(input, row);
            const auto shifted = ormi::core::ops::add(scaled, column);
            return ormi::core::ops::mean(
                ormi::core::ops::mul(shifted, shifted),
                {0, 2}
            );
        };

        auto warmup = build();
        warmup->realize();

        for (auto _ : state) {
            state.PauseTiming();
            auto output = build();
            state.ResumeTiming();

            output->realize();
            benchmark::DoNotOptimize(output->data());
        }

        state.SetItemsProcessed(
            state.iterations() * first * middle * last
        );
    }

    void bmPartialMapSumRealize(benchmark::State& state, bool force_materialized) {
        const auto rows = state.range(0);
        const auto columns = state.range(1);
        const auto input = tensor({rows, columns});

        auto warmup_mapped = broadcast_chain(input, input);
        auto warmup = ormi::core::ops::sum(warmup_mapped, {1});
        if (force_materialized) {
            warmup_mapped->realize();
        }
        warmup->realize();

        for (auto _ : state) {
            state.PauseTiming();
            auto mapped = broadcast_chain(input, input);
            auto output = ormi::core::ops::sum(mapped, {1});
            state.ResumeTiming();

            if (force_materialized) {
                mapped->realize();
            }
            output->realize();
            benchmark::DoNotOptimize(output->data());
        }

        state.SetItemsProcessed(state.iterations() * rows * columns);
    }

    void bmTransposedMapSumRealize(
        benchmark::State& state,
        bool force_materialized
    ) {
        const auto rows = state.range(0);
        const auto columns = state.range(1);
        const auto base = tensor({columns, rows});
        const auto input = ormi::core::ops::transpose(base, {1, 0});

        auto warmup_mapped = arithmetic_chain(input);
        auto warmup = ormi::core::ops::sum(warmup_mapped, {1});
        if (force_materialized) {
            warmup_mapped->realize();
        }
        warmup->realize();

        for (auto _ : state) {
            state.PauseTiming();
            auto mapped = arithmetic_chain(input);
            auto output = ormi::core::ops::sum(mapped, {1});
            state.ResumeTiming();

            if (force_materialized) {
                mapped->realize();
            }
            output->realize();
            benchmark::DoNotOptimize(output->data());
        }

        state.SetItemsProcessed(state.iterations() * rows * columns);
    }

    void bmTransposedGlobalSumRealize(benchmark::State& state) {
        const auto rows = state.range(0);
        const auto columns = state.range(1);
        const auto base = tensor({columns, rows});
        std::fill_n(
            static_cast<float*>(base->data()),
            base->numel(),
            0.1f
        );
        const auto input = ormi::core::ops::transpose(base, {1, 0});

        auto warmup = ormi::core::ops::sum(arithmetic_chain(input));
        warmup->realize();

        for (auto _ : state) {
            state.PauseTiming();
            auto output = ormi::core::ops::sum(arithmetic_chain(input));
            state.ResumeTiming();

            output->realize();
            benchmark::DoNotOptimize(output->data());
        }

        state.SetItemsProcessed(state.iterations() * rows * columns);
        const auto actual =
            static_cast<double>(*static_cast<const float*>(warmup->data()));
        const float value = 0.1f;
        const float mapped = (value * value + value) * value;
        const auto expected = static_cast<double>(mapped) *
            static_cast<double>(rows * columns);
        state.counters["relative_error_ppm"] =
            std::abs(actual - expected) / std::abs(expected) * 1.0e6;
    }

    void bmTransposedMappedGlobalSumRealize(benchmark::State& state) {
        const auto rows = state.range(0);
        const auto columns = state.range(1);
        const auto base = tensor({columns, rows});
        const auto input = ormi::core::ops::transpose(base, {1, 0});

        auto warmup = ormi::core::ops::sum(broadcast_chain(input, input));
        warmup->realize();

        for (auto _ : state) {
            state.PauseTiming();
            auto output = ormi::core::ops::sum(broadcast_chain(input, input));
            state.ResumeTiming();

            output->realize();
            benchmark::DoNotOptimize(output->data());
        }

        state.SetItemsProcessed(state.iterations() * rows * columns);
        const auto actual =
            static_cast<double>(*static_cast<const float*>(warmup->data()));
        const auto expected =
            static_cast<double>(std::sin(0.75f)) *
            static_cast<double>(rows * columns);
        state.counters["relative_error_ppm"] =
            std::abs(actual - expected) / std::abs(expected) * 1.0e6;
    }

    void bmExpandedGlobalSumRealize(benchmark::State& state) {
        const auto rows = state.range(0);
        const auto columns = state.range(1);
        const auto base = tensor({rows, 1});
        const auto input = ormi::core::ops::expand(base, {rows, columns});

        auto warmup = ormi::core::ops::sum(arithmetic_chain(input));
        warmup->realize();

        for (auto _ : state) {
            state.PauseTiming();
            auto output = ormi::core::ops::sum(arithmetic_chain(input));
            state.ResumeTiming();

            output->realize();
            benchmark::DoNotOptimize(output->data());
        }

        state.SetItemsProcessed(state.iterations() * rows * columns);
    }

    void bmMixedLayoutGlobalSumRealize(benchmark::State& state) {
        const auto rows = state.range(0);
        const auto columns = state.range(1);
        const auto base = tensor({columns, rows});
        const auto transposed =
            ormi::core::ops::transpose(base, {1, 0});
        const auto contiguous = tensor({rows, columns});

        auto build = [&] {
            const auto square =
                ormi::core::ops::mul(transposed, transposed);
            const auto added =
                ormi::core::ops::add(square, contiguous);
            return ormi::core::ops::sum(
                ormi::core::ops::mul(added, transposed)
            );
        };

        auto warmup = build();
        warmup->realize();

        for (auto _ : state) {
            state.PauseTiming();
            auto output = build();
            state.ResumeTiming();

            output->realize();
            benchmark::DoNotOptimize(output->data());
        }

        state.SetItemsProcessed(state.iterations() * rows * columns);
    }

    void bmPermutedRank3GlobalSumRealize(benchmark::State& state) {
        const auto dim0 = state.range(0);
        const auto dim1 = state.range(1);
        const auto dim2 = state.range(2);
        const auto base = tensor({dim1, dim2, dim0});
        const auto input =
            ormi::core::ops::transpose(base, {2, 0, 1});

        auto warmup =
            ormi::core::ops::sum(arithmetic_chain(input));
        warmup->realize();

        for (auto _ : state) {
            state.PauseTiming();
            auto output =
                ormi::core::ops::sum(arithmetic_chain(input));
            state.ResumeTiming();

            output->realize();
            benchmark::DoNotOptimize(output->data());
        }

        state.SetItemsProcessed(
            state.iterations() * dim0 * dim1 * dim2
        );
    }

    void bmMixedRank3GlobalSumRealize(benchmark::State& state) {
        const auto dim0 = state.range(0);
        const auto dim1 = state.range(1);
        const auto dim2 = state.range(2);
        const auto base = tensor({dim1, dim2, dim0});
        const auto permuted =
            ormi::core::ops::transpose(base, {2, 0, 1});
        const auto contiguous = tensor({dim0, dim1, dim2});

        auto build = [&] {
            const auto square =
                ormi::core::ops::mul(permuted, permuted);
            const auto added =
                ormi::core::ops::add(square, contiguous);
            return ormi::core::ops::sum(
                ormi::core::ops::mul(added, permuted)
            );
        };

        auto warmup = build();
        warmup->realize();

        for (auto _ : state) {
            state.PauseTiming();
            auto output = build();
            state.ResumeTiming();

            output->realize();
            benchmark::DoNotOptimize(output->data());
        }

        state.SetItemsProcessed(
            state.iterations() * dim0 * dim1 * dim2
        );
    }

    void bmPermutedRank3PartialSumRealize(benchmark::State& state) {
        const auto dim0 = state.range(0);
        const auto dim1 = state.range(1);
        const auto dim2 = state.range(2);
        const auto base = tensor({dim1, dim2, dim0});
        const auto input =
            ormi::core::ops::transpose(base, {2, 0, 1});

        auto warmup =
            ormi::core::ops::sum(arithmetic_chain(input), {1});
        warmup->realize();

        for (auto _ : state) {
            state.PauseTiming();
            auto output =
                ormi::core::ops::sum(arithmetic_chain(input), {1});
            state.ResumeTiming();

            output->realize();
            benchmark::DoNotOptimize(output->data());
        }

        state.SetItemsProcessed(
            state.iterations() * dim0 * dim1 * dim2
        );
    }

    void bmAutocastMapSumRealize(benchmark::State& state, bool force_materialized) {
        const auto elements = state.range(0);
        const auto input = tensor({elements}, ormi::core::DType::float16);

        auto build = [&] {
            const ormi::core::Autocast autocast;
            auto mapped = ormi::core::ops::sin(input);
            auto output = ormi::core::ops::sum(mapped);
            return std::pair{std::move(mapped), std::move(output)};
        };

        auto [warmup_mapped, warmup] = build();
        if (force_materialized) {
            warmup_mapped->realize();
        }
        warmup->realize();

        for (auto _ : state) {
            state.PauseTiming();
            auto [mapped, output] = build();
            state.ResumeTiming();

            if (force_materialized) {
                mapped->realize();
            }
            output->realize();
            benchmark::DoNotOptimize(output->data());
        }

        state.SetItemsProcessed(state.iterations() * elements);
    }

    void bmAutocastArithmeticReductionRealize(
        benchmark::State& state,
        Reduction reduction
    ) {
        const auto elements = state.range(0);
        const auto input = tensor({elements}, ormi::core::DType::float16);

        auto build = [&] {
            const ormi::core::Autocast autocast;
            return reduce(arithmetic_chain(input), reduction);
        };

        auto warmup = build();
        warmup->realize();

        for (auto _ : state) {
            state.PauseTiming();
            auto output = build();
            state.ResumeTiming();

            output->realize();
            benchmark::DoNotOptimize(output->data());
        }

        state.SetItemsProcessed(state.iterations() * elements);
    }

}

#define ORMI_FUSION_SHAPES(BENCHMARK_NAME) \
    BENCHMARK_NAME->Args({1, 1})->Args({32, 32})->Args({1024, 1024})

ORMI_FUSION_SHAPES(
    BENCHMARK_CAPTURE(bmBroadcastRealize, equal, InputShape::equal)
        ->Name("runtime/fusion/realize_equal")
);
ORMI_FUSION_SHAPES(
    BENCHMARK_CAPTURE(bmBroadcastRealize, scalar, InputShape::scalar)
        ->Name("runtime/fusion/realize_broadcast_scalar")
);
ORMI_FUSION_SHAPES(
    BENCHMARK_CAPTURE(bmBroadcastRealize, row, InputShape::row)
        ->Name("runtime/fusion/realize_broadcast_row")
);
ORMI_FUSION_SHAPES(
    BENCHMARK_CAPTURE(bmBroadcastRealize, column, InputShape::column)
        ->Name("runtime/fusion/realize_broadcast_column")
);
ORMI_FUSION_SHAPES(
    BENCHMARK_CAPTURE(bmUnfusedBroadcastRealize, scalar, InputShape::scalar)
        ->Name("runtime/fusion/unfused_broadcast_scalar")
);
ORMI_FUSION_SHAPES(
    BENCHMARK_CAPTURE(bmUnfusedBroadcastRealize, row, InputShape::row)
        ->Name("runtime/fusion/unfused_broadcast_row")
);
ORMI_FUSION_SHAPES(
    BENCHMARK_CAPTURE(bmUnfusedBroadcastRealize, column, InputShape::column)
        ->Name("runtime/fusion/unfused_broadcast_column")
);
ORMI_FUSION_SHAPES(
    BENCHMARK_CAPTURE(bmBroadcastArithmeticRealize, row, InputShape::row)
        ->Name("runtime/fusion/arithmetic_broadcast_row")
);
ORMI_FUSION_SHAPES(
    BENCHMARK_CAPTURE(bmBroadcastArithmeticRealize, column, InputShape::column)
        ->Name("runtime/fusion/arithmetic_broadcast_column")
);
ORMI_FUSION_SHAPES(
    BENCHMARK_CAPTURE(
        bmBroadcastArithmeticRealize,
        float16_row,
        InputShape::row,
        ormi::core::DType::float16
    )->Name("runtime/fusion/arithmetic_float16_broadcast_row")
);
ORMI_FUSION_SHAPES(
    BENCHMARK_CAPTURE(
        bmBroadcastArithmeticRealize,
        float16_column,
        InputShape::column,
        ormi::core::DType::float16
    )->Name("runtime/fusion/arithmetic_float16_broadcast_column")
);
ORMI_FUSION_SHAPES(
    BENCHMARK_CAPTURE(bmTransposeRealize, fused, true)
        ->Name("runtime/fusion/realize_transpose")
);
ORMI_FUSION_SHAPES(
    BENCHMARK_CAPTURE(bmTransposeRealize, unfused, false)
        ->Name("runtime/fusion/unfused_transpose")
);

BENCHMARK_CAPTURE(bmFloat16Realize, fused, false)
    ->Name("runtime/fusion/realize_float16")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576);
BENCHMARK_CAPTURE(bmFloat16Realize, unfused, true)
    ->Name("runtime/fusion/unfused_float16")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576);
BENCHMARK(bmFloat16ArithmeticRealize)
    ->Name("runtime/fusion/arithmetic_float16")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576);

BENCHMARK_CAPTURE(bmArithmeticRealize, fused, false)
    ->Name("runtime/fusion/arithmetic")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(65536)
    ->Arg(262144)
    ->Arg(524288)
    ->Arg(1048576);
BENCHMARK_CAPTURE(bmArithmeticRealize, materialized, true)
    ->Name("runtime/fusion/materialized_arithmetic")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576);
BENCHMARK(bmScalarArithmeticRealize)
    ->Name("runtime/fusion/scalar_arithmetic")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576);

BENCHMARK_CAPTURE(bmMapReductionRealize, sum_fused, Reduction::sum, false)
    ->Name("runtime/fusion/map_sum")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(262144)
    ->Arg(524288)
    ->Arg(1048576);
BENCHMARK_CAPTURE(bmMapReductionRealize, sum_materialized, Reduction::sum, true)
    ->Name("runtime/fusion/materialized_map_sum")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576);

BENCHMARK_CAPTURE(bmMapReductionRealize, mean_fused, Reduction::mean, false)
    ->Name("runtime/fusion/map_mean")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(262144)
    ->Arg(524288)
    ->Arg(1048576);
BENCHMARK_CAPTURE(bmMapReductionRealize, mean_materialized, Reduction::mean, true)
    ->Name("runtime/fusion/materialized_map_mean")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576);

BENCHMARK_CAPTURE(bmArithmeticMapReductionRealize, sum, Reduction::sum)
    ->Name("runtime/fusion/arithmetic_map_sum")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(262144)
    ->Arg(524288)
    ->Arg(1048576);
BENCHMARK_CAPTURE(bmArithmeticMapReductionRealize, mean, Reduction::mean)
    ->Name("runtime/fusion/arithmetic_map_mean")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(262144)
    ->Arg(524288)
    ->Arg(1048576);

BENCHMARK_CAPTURE(
    bmArithmeticMapReductionRealize,
    float16_sum,
    Reduction::sum,
    ormi::core::DType::float16
)
    ->Name("runtime/fusion/arithmetic_float16_map_sum")
    ->Arg(1024)
    ->Arg(262144)
    ->Arg(524288)
    ->Arg(1048576);
BENCHMARK_CAPTURE(
    bmArithmeticMapReductionRealize,
    float16_mean,
    Reduction::mean,
    ormi::core::DType::float16
)
    ->Name("runtime/fusion/arithmetic_float16_map_mean")
    ->Arg(1024)
    ->Arg(262144)
    ->Arg(524288)
    ->Arg(1048576);

BENCHMARK_CAPTURE(
    bmPartialArithmeticReductionRealize,
    sum,
    Reduction::sum,
    1
)
    ->Name("runtime/fusion/arithmetic_last_axis_sum")
    ->Args({1, 1})
    ->Args({32, 32})
    ->Args({1024, 1024});
BENCHMARK_CAPTURE(
    bmPartialArithmeticReductionRealize,
    mean,
    Reduction::mean,
    1
)
    ->Name("runtime/fusion/arithmetic_last_axis_mean")
    ->Args({1, 1})
    ->Args({32, 32})
    ->Args({1024, 1024});
BENCHMARK_CAPTURE(
    bmPartialArithmeticReductionRealize,
    float16_sum,
    Reduction::sum,
    1,
    ormi::core::DType::float16
)
    ->Name("runtime/fusion/arithmetic_float16_last_axis_sum")
    ->Args({32, 32})
    ->Args({32, 64})
    ->Args({1024, 1024});
BENCHMARK_CAPTURE(
    bmPartialArithmeticReductionRealize,
    float16_mean,
    Reduction::mean,
    1,
    ormi::core::DType::float16
)
    ->Name("runtime/fusion/arithmetic_float16_last_axis_mean")
    ->Args({32, 32})
    ->Args({32, 64})
    ->Args({1024, 1024});

BENCHMARK_CAPTURE(
    bmPartialArithmeticReductionRealize,
    first_axis_sum,
    Reduction::sum,
    0
)
    ->Name("runtime/fusion/arithmetic_first_axis_sum")
    ->Args({32, 32})
    ->Args({64, 64})
    ->Args({128, 128})
    ->Args({256, 256})
    ->Args({512, 512})
    ->Args({1024, 1024});

BENCHMARK(bmSeparatedAxesMeanRealize)
    ->Name("runtime/fusion/separated_axes_mean")
    ->Args({4, 8, 16})
    ->Args({32, 16, 64})
    ->Args({256, 64, 64});

BENCHMARK_CAPTURE(bmPartialMapSumRealize, fused, false)
    ->Name("runtime/fusion/partial_map_sum")
    ->Args({1, 1})
    ->Args({32, 32})
    ->Args({1024, 1024});
BENCHMARK_CAPTURE(bmPartialMapSumRealize, materialized, true)
    ->Name("runtime/fusion/materialized_partial_map_sum")
    ->Args({1, 1})
    ->Args({32, 32})
    ->Args({1024, 1024});

BENCHMARK_CAPTURE(bmPartialReductionCall, direct, false)
    ->Name("runtime/fusion/partial_reduction_call")
    ->Arg(1048576)
    ->Arg(4194304)
    ->UseRealTime();
BENCHMARK_CAPTURE(bmPartialReductionCall, prepared, true)
    ->Name("runtime/fusion/prepared_partial_reduction_call")
    ->Arg(1048576)
    ->Arg(4194304)
    ->UseRealTime();
BENCHMARK_CAPTURE(bmVectorizedReductionCall, direct, false)
    ->Name("runtime/fusion/vectorized_reduction_call")
    ->Arg(1048576)
    ->Arg(4194304)
    ->UseRealTime();
BENCHMARK_CAPTURE(bmVectorizedReductionCall, prepared, true)
    ->Name("runtime/fusion/prepared_vectorized_reduction_call")
    ->Arg(1048576)
    ->Arg(4194304)
    ->UseRealTime();
BENCHMARK_CAPTURE(bmPackedReductionCall, direct, false)
    ->Name("runtime/fusion/packed_reduction_call")
    ->Arg(1048576)
    ->Arg(4194304)
    ->UseRealTime();
BENCHMARK_CAPTURE(bmPackedReductionCall, prepared, true)
    ->Name("runtime/fusion/prepared_packed_reduction_call")
    ->Arg(1048576)
    ->Arg(4194304)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmTransposedMapSumRealize, fused, false)
    ->Name("runtime/fusion/transposed_map_sum")
    ->Args({32, 32})
    ->Args({64, 64})
    ->Args({128, 128})
    ->Args({256, 256})
    ->Args({512, 512})
    ->Args({1024, 1024});
BENCHMARK_CAPTURE(bmTransposedMapSumRealize, materialized, true)
    ->Name("runtime/fusion/materialized_transposed_map_sum")
    ->Args({32, 32})
    ->Args({64, 64})
    ->Args({128, 128})
    ->Args({256, 256})
    ->Args({512, 512})
    ->Args({1024, 1024});

BENCHMARK(bmTransposedGlobalSumRealize)
    ->Name("runtime/fusion/transposed_global_sum")
    ->Args({512, 512})
    ->Args({1024, 1024})
    ->Args({32, 32768});
BENCHMARK(bmTransposedMappedGlobalSumRealize)
    ->Name("runtime/fusion/transposed_mapped_global_sum")
    ->Args({1024, 1024})
    ->UseRealTime();
BENCHMARK(bmExpandedGlobalSumRealize)
    ->Name("runtime/fusion/expanded_global_sum")
    ->Args({512, 512})
    ->Args({1024, 1024})
    ->Args({32, 32768})
    ->Args({1, 1048576});
BENCHMARK(bmMixedLayoutGlobalSumRealize)
    ->Name("runtime/fusion/mixed_layout_global_sum")
    ->Args({1024, 1024})
    ->Args({32, 32768});
BENCHMARK(bmPermutedRank3GlobalSumRealize)
    ->Name("runtime/fusion/permuted_rank3_global_sum")
    ->Args({64, 64, 64})
    ->Args({32, 128, 256});
BENCHMARK(bmMixedRank3GlobalSumRealize)
    ->Name("runtime/fusion/mixed_rank3_global_sum")
    ->Args({64, 64, 64})
    ->Args({32, 128, 256});
BENCHMARK(bmPermutedRank3PartialSumRealize)
    ->Name("runtime/fusion/permuted_rank3_middle_axis_sum")
    ->Args({64, 64, 64})
    ->Args({32, 128, 256});

BENCHMARK_CAPTURE(bmAutocastMapSumRealize, fused, false)
    ->Name("runtime/fusion/autocast_map_sum")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576);
BENCHMARK_CAPTURE(bmAutocastMapSumRealize, materialized, true)
    ->Name("runtime/fusion/materialized_autocast_map_sum")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576);

BENCHMARK_CAPTURE(
    bmAutocastArithmeticReductionRealize,
    sum,
    Reduction::sum
)
    ->Name("runtime/fusion/autocast_arithmetic_sum")
    ->Arg(1024)
    ->Arg(1048576);
BENCHMARK_CAPTURE(
    bmAutocastArithmeticReductionRealize,
    mean,
    Reduction::mean
)
    ->Name("runtime/fusion/autocast_arithmetic_mean")
    ->Arg(1024)
    ->Arg(1048576);

#undef ORMI_FUSION_SHAPES
