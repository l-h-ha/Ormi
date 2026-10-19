#include <cstddef>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include <ormi/core/setup.h>
#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace {
    using ormi::core::Setup;
    using ormi::core::Tensor;

    static_assert(!std::is_copy_constructible_v<Setup>);
    static_assert(!std::is_copy_assignable_v<Setup>);
    static_assert(std::is_nothrow_move_constructible_v<Setup>);
    static_assert(std::is_nothrow_move_assignable_v<Setup>);

    struct Counts {
        std::size_t runs = 0;
        std::size_t drops = 0;
        Tensor* output = nullptr;
        bool fail = false;
    };

    struct State {
        Counts& counts;
    };

    Setup prepare(Counts& counts) {
        return Setup(
            new State{counts},
            [](void* state, Tensor& output) {
                auto& counts = static_cast<State*>(state)->counts;
                ++counts.runs;
                counts.output = &output;
                if (counts.fail) {
                    ORMI_THROW(ormi::OperationError, "Prepared execution failed.");
                }
            },
            [](void* state) noexcept {
                auto* owned = static_cast<State*>(state);
                ++owned->counts.drops;
                delete owned;
            },
            2,
            64
        );
    }

    void expect_empty(const Setup& setup) {
        EXPECT_FALSE(setup);
        EXPECT_EQ(setup.buffers(), 0);
        EXPECT_EQ(setup.bytes(), 0);
    }

    TEST(SetupTest, startsEmpty) {
        const ormi::core::Setup setup;
        expect_empty(setup);
    }

    TEST(SetupTest, releasesUnrunState) {
        Counts counts;
        {
            const auto setup = prepare(counts);
            EXPECT_TRUE(setup);
            EXPECT_EQ(setup.buffers(), 2);
            EXPECT_EQ(setup.bytes(), 64);
        }
        EXPECT_EQ(counts.runs, 0);
        EXPECT_EQ(counts.drops, 1);
    }

    TEST(SetupTest, transfersOwnershipAndMetricsDuringMoveConstruction) {
        Counts counts;
        {
            auto original = prepare(counts);
            const auto moved = std::move(original);
            expect_empty(original);
            EXPECT_TRUE(moved);
            EXPECT_EQ(moved.buffers(), 2);
            EXPECT_EQ(moved.bytes(), 64);
            EXPECT_EQ(counts.drops, 0);
        }
        EXPECT_EQ(counts.drops, 1);
    }

    TEST(SetupTest, moveAssignmentReleasesPreviousState) {
        Counts previous;
        Counts next;
        {
            auto destination = prepare(previous);
            auto source = prepare(next);
            destination = std::move(source);
            expect_empty(source);
            EXPECT_TRUE(destination);
            EXPECT_EQ(destination.buffers(), 2);
            EXPECT_EQ(destination.bytes(), 64);
            EXPECT_EQ(previous.drops, 1);
            EXPECT_EQ(next.drops, 0);
        }
        EXPECT_EQ(previous.drops, 1);
        EXPECT_EQ(next.drops, 1);
    }

    TEST(SetupTest, assigningEmptyStateReleasesOwnership) {
        Counts counts;
        {
            auto setup = prepare(counts);
            setup = {};
            expect_empty(setup);
            EXPECT_EQ(counts.drops, 1);
        }
        EXPECT_EQ(counts.drops, 1);
    }

    TEST(SetupTest, selfMovePreservesOwnership) {
        Counts counts;
        {
            auto setup = prepare(counts);
            auto& same = setup;
            setup = std::move(same);
            EXPECT_TRUE(setup);
            EXPECT_EQ(setup.buffers(), 2);
            EXPECT_EQ(setup.bytes(), 64);
            EXPECT_EQ(counts.drops, 0);
        }
        EXPECT_EQ(counts.drops, 1);
    }

    TEST(SetupTest, invokesTheRetainedStateWithTheProvidedOutput) {
        Tensor output(ormi::core::Device::cpu(), {}, ormi::core::DType::float32);
        Counts counts;
        {
            const auto setup = prepare(counts);
            setup.run(output);
            setup.run(output);
            EXPECT_EQ(counts.runs, 2);
            EXPECT_EQ(counts.output, &output);
            EXPECT_EQ(counts.drops, 0);
        }
        EXPECT_EQ(counts.drops, 1);
    }

    TEST(SetupTest, keepsOwnershipAfterExecutionFailure) {
        Tensor output(ormi::core::Device::cpu(), {}, ormi::core::DType::float32);
        Counts counts;
        {
            const auto setup = prepare(counts);
            counts.fail = true;
            EXPECT_THROW(setup.run(output), ormi::OperationError);
            EXPECT_TRUE(setup);
            EXPECT_EQ(counts.drops, 0);
            counts.fail = false;
            setup.run(output);
            EXPECT_EQ(counts.runs, 2);
        }
        EXPECT_EQ(counts.drops, 1);
    }
}
