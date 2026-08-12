#include <array>
#include <atomic>
#include <barrier>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/jit/compile.h"
#include "backends/cpu/jit/loop.h"
#include "core/runtime/fusion/expr.h"

namespace {
    constexpr std::size_t elements = 32 * 1024;

    std::shared_ptr<ormi::core::Tensor> input() {
        return std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{static_cast<std::int64_t>(elements)},
            ormi::core::DType::float32
        );
    }

    std::size_t work(const std::shared_ptr<ormi::core::Tensor>& output) {
        const auto expression = ormi::core::runtime::fusion::build(*output);
        EXPECT_TRUE(expression.has_value());
        return ormi::backends::cpu::jit::parallel_work(*expression, false);
    }

    struct Graph {
        std::shared_ptr<ormi::core::Tensor> root;
        ormi::core::runtime::fusion::Expr expression;
    };

    Graph graph(std::size_t additions) {
        const auto x = input();
        auto root = ormi::core::ops::mul(x, x);
        for (std::size_t i = 0; i < additions; ++i) {
            root = ormi::core::ops::add(root, x);
        }
        root = ormi::core::ops::sin(root);
        auto expression = ormi::core::runtime::fusion::build(*root);
        if (!expression) {
            throw std::logic_error("Cannot build CPU JIT compile test expression.");
        }
        return Graph{std::move(root), std::move(*expression)};
    }

    TEST(CPUJITCompileTest, weightsOnlyTranscendentalOperations) {
        const auto x = input();
        const auto added = ormi::core::ops::add(x, x);

        EXPECT_EQ(
            work(ormi::core::ops::mul(added, x)),
            elements
        );
        EXPECT_EQ(
            work(ormi::core::ops::neg(added)),
            elements
        );
        EXPECT_EQ(
            work(ormi::core::ops::sin(added)),
            16 * elements
        );
        EXPECT_EQ(
            work(ormi::core::ops::cos(ormi::core::ops::sin(x))),
            31 * elements
        );

        const auto sine = ormi::core::ops::sin(added);
        const auto cosine = ormi::core::ops::cos(added);
        const std::array<ormi::core::Tensor*, 1> auxiliaries{cosine.get()};
        const std::unordered_set<const ormi::core::Tensor*> boundaries;
        const auto paired = ormi::core::runtime::fusion::build(
            *sine,
            auxiliaries,
            boundaries
        );
        ASSERT_TRUE(paired.has_value());
        EXPECT_EQ(
            ormi::backends::cpu::jit::parallel_work(*paired, false),
            31 * elements
        );
        EXPECT_EQ(
            ormi::backends::cpu::jit::parallel_work(*paired, true),
            16 * elements
        );
    }

    TEST(CPUJITCompileTest, findsKernelPublishedByAnotherThread) {
        const auto first = graph(11);
        const auto second = graph(13);
        const auto expected_first = ormi::backends::cpu::jit::compile(
            first.expression
        );
        std::shared_ptr<ormi::backends::cpu::jit::Compiled> actual_first;
        std::shared_ptr<ormi::backends::cpu::jit::Compiled> actual_second;
        std::barrier synchronized(2);

        std::jthread worker([&] {
            actual_first = ormi::backends::cpu::jit::compile(first.expression);
            synchronized.arrive_and_wait();
            synchronized.arrive_and_wait();
            actual_second = ormi::backends::cpu::jit::compile(second.expression);
        });

        synchronized.arrive_and_wait();
        const auto expected_second = ormi::backends::cpu::jit::compile(
            second.expression
        );
        synchronized.arrive_and_wait();
        worker.join();

        EXPECT_EQ(actual_first.get(), expected_first.get());
        EXPECT_EQ(actual_second.get(), expected_second.get());
    }

    TEST(CPUJITCompileTest, reusesAlternatingKernelsAcrossThreads) {
        const auto first = graph(11);
        const auto second = graph(13);
        const auto expected_first = ormi::backends::cpu::jit::compile(
            first.expression
        );
        const auto expected_second = ormi::backends::cpu::jit::compile(
            second.expression
        );
        std::atomic<bool> correct{true};
        std::vector<std::jthread> workers;
        workers.reserve(8);

        while (workers.size() < 8) {
            workers.emplace_back([&] {
                for (std::size_t i = 0; i < 1'000; ++i) {
                    const bool first_kernel = i % 2 == 0;
                    const auto compiled = ormi::backends::cpu::jit::compile(
                        first_kernel ? first.expression : second.expression
                    );
                    const auto* expected = first_kernel
                        ? expected_first.get()
                        : expected_second.get();
                    if (compiled.get() != expected) {
                        correct.store(false, std::memory_order_relaxed);
                    }
                }
            });
        }
        workers.clear();

        EXPECT_TRUE(correct.load(std::memory_order_relaxed));
    }

    TEST(CPUJITCompileTest, loadsSeveralKernelsFromOneLibrary) {
        const auto first = graph(17);
        const auto second = graph(19);
        const std::array<const ormi::core::runtime::fusion::Expr*, 2> expressions{
            &first.expression,
            &second.expression
        };

        const auto compiled = ormi::backends::cpu::jit::compile(expressions);

        ASSERT_EQ(compiled.size(), 2);
        EXPECT_EQ(compiled[0]->library.get(), compiled[1]->library.get());
        EXPECT_NE(compiled[0]->function, compiled[1]->function);
        EXPECT_EQ(
            compiled[0].get(),
            ormi::backends::cpu::jit::compile(first.expression).get()
        );
        EXPECT_EQ(
            compiled[1].get(),
            ormi::backends::cpu::jit::compile(second.expression).get()
        );
    }
}
