#include <cstddef>
#include <cstdint>
#include <vector>

#include <benchmark/benchmark.h>

#include "backends/cpu/executor.h"

namespace {
    struct alignas(64) Counter {
        std::uint64_t value = 0;
    };

    struct Counters {
        std::vector<Counter> values;
    };

    void count_submission(
        void* opaque,
        std::size_t slot,
        std::size_t,
        std::size_t
    ) {
        auto& counters = *static_cast<Counters*>(opaque);
        ++counters.values[slot].value;
    }

    void bmCPUExecutorSubmit(benchmark::State& state) {
        constexpr std::size_t range = 1024;
        constexpr std::size_t unit = 1;
        constexpr std::size_t work = 512 * 1024;
        const auto slots = ormi::backends::cpu::executor::slots(
            range,
            unit,
            work
        );
        Counters counters{std::vector<Counter>(slots)};

        for (auto _ : state) {
            ormi::backends::cpu::executor::run(
                range,
                unit,
                work,
                count_submission,
                &counters
            );
            benchmark::ClobberMemory();
        }

        state.counters["participants"] = static_cast<double>(slots);
    }
}

BENCHMARK(bmCPUExecutorSubmit)
    ->Name("runtime/cpu_executor/submit")
    ->UseRealTime();
