#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <ormi/core/autograd.h>
#include <ormi/core/memory/ops.h>
#include <ormi/core/ops/cast.h>
#include <ormi/core/ops/index.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/plan.h>

namespace {
    TEST(CUDASelectTest, scattersAcrossBlocksAndReadsTheCurrentScalarDuringReplay) {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
            GTEST_SKIP() << "No CUDA device is available.";
        }

        using namespace ormi::core;
        const auto device = Device::from("cuda");
        const auto input = std::make_shared<Tensor>(device, Shape{2051}, DType::float64);
        const auto seed_values = std::make_shared<Tensor>(device, Shape{2, 2}, DType::float64);
        std::array values{1., 2., 7., 4.};
        memory::copy(seed_values->data(), device, values.data(), Device::cpu(), sizeof(values));
        const auto seed = ops::select(ops::transpose(seed_values, {1, 0}), 1);
        const auto gradient = grad(ops::select(input, 2050), input, seed);
        Plan plan({gradient});
        plan.run();
        auto replay = plan.record();
        for (const double value : {7., -3.}) {
            values[2] = value;
            memory::copy(seed_values->data(), device, values.data(), Device::cpu(), sizeof(values));
            replay.run();
            std::vector<double> actual(input->numel());
            memory::copy(actual.data(), Device::cpu(), gradient->data(), device, gradient->bytes());
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_DOUBLE_EQ(actual[i], i == 2050 ? value : 0.);
            }
        }
    }

    TEST(CUDASelectTest, preservesStridesAdjointsAndStorageAcrossReplay) {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
            GTEST_SKIP() << "No CUDA device is available.";
        }

        using namespace ormi::core;
        const auto device = Device::from("cuda");
        const std::array values{1., 2., 3., 4., 5., 6.};
        for (const auto dtype : {DType::float16, DType::float32, DType::float64}) {
            auto base = std::make_shared<Tensor>(device, Shape{6}, DType::float64);
            memory::copy(base->data(), device, values.data(), Device::cpu(), sizeof(values));
            auto input = ops::cast(base, dtype);
            input->realize();
            auto view = std::make_shared<Tensor>(
                input->buffer(),
                Shape{2, 2},
                dtype,
                std::vector<std::int64_t>{3, 1},
                dtype_size(dtype)
            );
            auto transposed = ops::transpose(view, {1, 0});
            auto selected = ops::select(transposed, 3);
            auto cube = ops::mul(ops::mul(selected, selected), selected);
            auto first = grad(cube, view);
            auto second = grad(ops::sum(first), view);
            auto third = grad(ops::sum(second), view);
            auto selected_value = ops::cast(selected, DType::float64);
            auto first_values = ops::cast(first, DType::float64);
            auto second_values = ops::cast(second, DType::float64);
            auto third_values = ops::cast(third, DType::float64);
            EXPECT_FALSE(selected->realized());
            EXPECT_FALSE(first->realized());
            auto plan =
                std::make_unique<Plan>(std::vector{selected_value, first_values, second_values, third_values}
                );
            plan->run();
            auto replay = plan->record();
            plan.reset();
            base.reset();
            input.reset();
            view.reset();
            transposed.reset();
            selected.reset();
            cube.reset();
            for (int repeat = 0; repeat < 3; ++repeat) {
                replay.run();
                double scalar = 0.;
                memory::copy(&scalar, Device::cpu(), selected_value->data(), device, sizeof(scalar));
                EXPECT_DOUBLE_EQ(scalar, 6.);
                const std::array roots{first_values, second_values, third_values};
                const std::array derivatives{108., 36., 6.};
                for (std::size_t order = 0; order < roots.size(); ++order) {
                    std::array<double, 4> actual{};
                    memory::copy(actual.data(), Device::cpu(), roots[order]->data(), device, sizeof(actual));
                    for (std::size_t i = 0; i < actual.size(); ++i) {
                        EXPECT_DOUBLE_EQ(actual[i], i == 3 ? derivatives[order] : 0.);
                    }
                }
            }
        }
    }
}
