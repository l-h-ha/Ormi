#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <string_view>

#include <ormi/core/error.h>

#include "backends/cpu/executor.h"
#include "backends/cpu/executor/pool.h"
#include "platform/platform.h"

namespace ormi::backends::cpu::executor {
    namespace {
        constexpr std::size_t parallel_threshold = 512 * 1024;
        constexpr std::size_t work_per_participant = 192 * 1024;
        std::size_t configured_threads() {
            const char* value = std::getenv("ORMI_CPU_THREADS");
            if (!value || !*value) {
                return platform::cpu_cores();
            }

            std::size_t result = 0;
            const std::string_view text(value);
            const auto parsed = std::from_chars(
                text.data(),
                text.data() + text.size(),
                result
            );
            if (
                parsed.ec != std::errc{} ||
                parsed.ptr != text.data() + text.size() ||
                result == 0
            ) {
                ORMI_THROW(
                    OperationError,
                    "ORMI_CPU_THREADS must be a positive integer."
                );
            }
            return result;
        }

        std::size_t thread_count() {
            static const std::size_t value = configured_threads();
            return value;
        }

        Pool& pool() {
            // A DLL static destructor cannot safely join worker threads while
            // Windows holds the loader lock during process shutdown.
            static Pool* value = new Pool(thread_count());
            return *value;
        }
    }

    std::size_t slots(
        std::size_t range,
        std::size_t unit,
        std::size_t work
    ) {
        if (unit == 0) {
            ORMI_THROW(
                OperationError,
                "CPU execution partition unit must be positive."
            );
        }
        if (work < parallel_threshold) {
            return 1;
        }
        const auto units = range / unit +
            static_cast<std::size_t>(range % unit != 0);
        if (units < 2 || thread_count() <= 1) {
            return 1;
        }

        const auto useful = work / work_per_participant +
            static_cast<std::size_t>(work % work_per_participant != 0);
        return std::min({
            thread_count(),
            units,
            std::max(std::size_t{2}, useful)
        });
    }

    void run(
        std::size_t range,
        std::size_t unit,
        std::size_t work,
        Function function,
        void* context
    ) {
        const auto participants = slots(range, unit, work);
        if (participants == 1) {
            function(context, 0, 0, range);
            return;
        }
        pool().run(range, unit, participants, function, context);
    }
}
