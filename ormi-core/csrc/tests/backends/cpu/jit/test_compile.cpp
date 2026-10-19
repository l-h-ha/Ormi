#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <span>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/jit/compile.h"
#include "backends/cpu/jit/execute.h"
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

    Graph graph(std::size_t additions, float value = 0.25f) {
        const auto x = input();
        std::fill_n(static_cast<float*>(x->data()), elements, value);
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

    TEST(CPUJITCompileTest, acceptsEmptyBatch) {
        const std::span<const ormi::core::runtime::fusion::Expr* const> expressions;

        EXPECT_TRUE(ormi::backends::cpu::jit::compile(expressions).empty());
    }

    TEST(CPUJITCompileTest, rejectsNullBatchEntry) {
        const std::array<const ormi::core::runtime::fusion::Expr*, 1> expressions{nullptr};

        EXPECT_THROW(ormi::backends::cpu::jit::compile(expressions), ormi::OperationError);
    }

    TEST(CPUJITCompileTest, preservesRequestOrderAndDeduplicatesEquivalentGraphs) {
        const auto first = graph(23, 0.25f);
        const auto second = graph(29, 0.5f);
        const auto equivalent = graph(23, 1.0f);
        const std::array<const ormi::core::runtime::fusion::Expr*, 4> expressions{
            &second.expression, &first.expression, &equivalent.expression, &second.expression
        };

        const auto compiled = ormi::backends::cpu::jit::compile(expressions);

        ASSERT_EQ(compiled.size(), expressions.size());
        EXPECT_EQ(compiled[0].get(), compiled[3].get());
        EXPECT_EQ(compiled[1].get(), compiled[2].get());
        EXPECT_NE(compiled[0].get(), compiled[1].get());
        EXPECT_EQ(compiled[0]->library.get(), compiled[1]->library.get());
        const std::array expected{
            std::sin(0.5f * 0.5f + 29.0f * 0.5f),
            std::sin(0.25f * 0.25f + 23.0f * 0.25f),
            std::sin(1.0f + 23.0f),
            std::sin(0.5f * 0.5f + 29.0f * 0.5f)
        };
        for (std::size_t i = 0; i < expressions.size(); ++i) {
            ormi::core::Tensor output(
                ormi::core::Device::cpu(), {static_cast<std::int64_t>(elements)},
                ormi::core::DType::float32
            );
            ormi::backends::cpu::jit::execution::run(output, {}, *expressions[i], *compiled[i]);
            const auto* values = static_cast<const float*>(output.data());
            for (std::size_t element = 0; element < elements; ++element) {
                ASSERT_NEAR(values[element], expected[i], 1e-5f);
            }
        }
    }

    TEST(CPUJITCompileTest, retainsMemoryHitsWhileCompilingNewBatchEntries) {
        const auto warm = graph(31);
        const auto first = graph(37);
        const auto second = graph(41);
        const auto expected = ormi::backends::cpu::jit::compile(warm.expression);
        const std::array<const ormi::core::runtime::fusion::Expr*, 4> expressions{
            &first.expression, &warm.expression, &second.expression, &first.expression
        };

        const auto compiled = ormi::backends::cpu::jit::compile(expressions);

        ASSERT_EQ(compiled.size(), expressions.size());
        EXPECT_EQ(compiled[1].get(), expected.get());
        EXPECT_EQ(compiled[0].get(), compiled[3].get());
        EXPECT_EQ(compiled[0]->library.get(), compiled[2]->library.get());
        EXPECT_NE(compiled[0]->function, compiled[2]->function);
        EXPECT_EQ(ormi::backends::cpu::jit::compile(warm.expression).get(), expected.get());
    }

    TEST(CPUJITCompileTest, publishesOneResultForConcurrentBatchMisses) {
        constexpr std::size_t workers = 4;
        const auto first = graph(43);
        const auto second = graph(47);
        const std::array<const ormi::core::runtime::fusion::Expr*, 3> expressions{
            &first.expression, &second.expression, &first.expression
        };
        std::array<std::vector<std::shared_ptr<ormi::backends::cpu::jit::Compiled>>, workers> results;
        std::array<std::exception_ptr, workers> errors{};
        std::barrier start(static_cast<std::ptrdiff_t>(workers));
        std::vector<std::jthread> threads;
        for (std::size_t worker = 0; worker < workers; ++worker) {
            threads.emplace_back([&, worker] {
                start.arrive_and_wait();
                try {
                    results[worker] = ormi::backends::cpu::jit::compile(expressions);
                } catch (...) {
                    errors[worker] = std::current_exception();
                }
            });
        }
        threads.clear();

        for (const auto& error : errors) {
            ASSERT_FALSE(error);
        }
        for (const auto& result : results) {
            ASSERT_EQ(result.size(), expressions.size());
            EXPECT_EQ(result[0].get(), results[0][0].get());
            EXPECT_EQ(result[1].get(), results[0][1].get());
            EXPECT_EQ(result[2].get(), result[0].get());
            EXPECT_EQ(result[0]->library.get(), result[1]->library.get());
        }
    }
}
