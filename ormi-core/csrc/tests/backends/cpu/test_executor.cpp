#include <atomic>
#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include "backends/cpu/executor.h"

namespace {
    struct Visits {
        std::vector<std::atomic<unsigned char>> values;
        std::size_t unit;
        std::atomic<bool> aligned{true};
    };

    void visit(
        void* opaque,
        std::size_t,
        std::size_t begin,
        std::size_t end
    ) {
        auto& visits = *static_cast<Visits*>(opaque);
        if (
            begin % visits.unit != 0 ||
            (end != visits.values.size() && end % visits.unit != 0)
        ) {
            visits.aligned = false;
        }
        for (std::size_t i = begin; i < end; ++i) {
            ++visits.values[i];
        }
    }

    struct Nested {
        std::atomic<std::size_t> calls{0};
        std::atomic<std::size_t> begin{0};
        std::atomic<std::size_t> end{0};
    };

    struct Calls {
        explicit Calls(std::size_t slots)
            : values(slots) {}

        std::vector<std::atomic<unsigned char>> values;
    };

    void count_call(
        void* opaque,
        std::size_t slot,
        std::size_t,
        std::size_t
    ) {
        auto& calls = *static_cast<Calls*>(opaque);
        ++calls.values[slot];
    }

    void nested_visit(
        void* opaque,
        std::size_t,
        std::size_t begin,
        std::size_t end
    ) {
        auto& nested = *static_cast<Nested*>(opaque);
        ++nested.calls;
        nested.begin = begin;
        nested.end = end;
    }

    void submit_nested(
        void* opaque,
        std::size_t slot,
        std::size_t,
        std::size_t
    ) {
        if (slot != 0) {
            return;
        }
        ormi::backends::cpu::executor::run(
            1024,
            1,
            512 * 1024,
            nested_visit,
            opaque
        );
    }

    TEST(CPUExecutorTest, visitsCompleteUnitsOnce) {
        constexpr std::size_t unit = 17;
        Visits visits{
            std::vector<std::atomic<unsigned char>>(unit * 30841),
            unit
        };

        ormi::backends::cpu::executor::run(
            visits.values.size(),
            visits.unit,
            visits.values.size(),
            visit,
            &visits
        );

        EXPECT_TRUE(visits.aligned);
        for (const auto& count : visits.values) {
            EXPECT_EQ(count.load(), 1);
        }
    }

    TEST(CPUExecutorTest, preservesFinalPartialUnit) {
        constexpr std::size_t unit = 17;
        Visits visits{
            std::vector<std::atomic<unsigned char>>(unit * 30841 + 5),
            unit
        };

        ormi::backends::cpu::executor::run(
            visits.values.size(),
            visits.unit,
            visits.values.size(),
            visit,
            &visits
        );

        EXPECT_TRUE(visits.aligned);
        for (const auto& count : visits.values) {
            EXPECT_EQ(count.load(), 1);
        }
    }

    TEST(CPUExecutorTest, usesEverySelectedSlotForUnevenUnits) {
        constexpr std::size_t unit = 4;
        constexpr std::size_t range = unit * 4;
        constexpr std::size_t work = 512 * 1024;
        const auto slots = ormi::backends::cpu::executor::slots(
            range,
            unit,
            work
        );
        if (slots < 3) {
            GTEST_SKIP() << "This check requires at least three CPU threads.";
        }
        Calls calls(slots);

        ormi::backends::cpu::executor::run(
            range,
            unit,
            work,
            count_call,
            &calls
        );

        for (const auto& count : calls.values) {
            EXPECT_EQ(count.load(), 1);
        }
    }

    TEST(CPUExecutorTest, runsNestedSubmissionSerially) {
        Nested nested;

        ormi::backends::cpu::executor::run(
            1024,
            1,
            512 * 1024,
            submit_nested,
            &nested
        );

        EXPECT_EQ(nested.calls.load(), 1u);
        EXPECT_EQ(nested.begin.load(), 0u);
        EXPECT_EQ(nested.end.load(), 1024u);
    }
}
