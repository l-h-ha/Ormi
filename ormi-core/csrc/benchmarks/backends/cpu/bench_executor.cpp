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

    void bmCPUExecutorSubmit(benchmark::State& state, std::size_t work) {
        constexpr std::size_t range = 1024;
        constexpr std::size_t unit = 1;
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

        for (const auto& counter : counters.values) {
            if (counter.value != static_cast<std::uint64_t>(state.iterations())) {
                state.SkipWithError("Executor did not visit every selected slot exactly once.");
                break;
            }
        }
        state.counters["participants"] = static_cast<double>(slots);
    }
}

BENCHMARK_CAPTURE(bmCPUExecutorSubmit, parallel, 512 * 1024)
    ->Name("runtime/cpu_executor/submit")
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCPUExecutorSubmit, serial, 0)
    ->Name("runtime/cpu_executor/serial_submit")
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCPUExecutorSubmit, full, 6 * 192 * 1024)
    ->Name("runtime/cpu_executor/full_submit")
    ->UseRealTime();
