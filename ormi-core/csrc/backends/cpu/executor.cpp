#include <algorithm>
#include <atomic>
#include <charconv>
#include <condition_variable>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string_view>
#include <thread>
#include <vector>

#include <ormi/core/error.h>

#include "backends/cpu/executor.h"
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

        std::size_t partition_begin(
            std::size_t range,
            std::size_t count,
            std::size_t unit,
            std::size_t index
        ) {
            const auto units = range / unit +
                static_cast<std::size_t>(range % unit != 0);
            const auto base = units / count;
            const auto extra = units % count;
            return (index * base + std::min(index, extra)) * unit;
        }

        class Pool {
        public:
            explicit Pool(std::size_t thread_count) {
                try {
                    workers_.reserve(thread_count - 1);
                    ready_.reserve(thread_count - 1);
                    for (std::size_t i = 1; i < thread_count; ++i) {
                        ready_.push_back(
                            std::make_unique<std::condition_variable>()
                        );
                    }
                    for (std::size_t i = 1; i < thread_count; ++i) {
                        workers_.emplace_back([this, i] {
                            worker(i);
                        });
                    }
                } catch (...) {
                    stop();
                    throw;
                }
            }

            ~Pool() {
                stop();
            }

            Pool(const Pool&) = delete;
            Pool& operator=(const Pool&) = delete;

            void run(
                std::size_t range,
                std::size_t unit,
                std::size_t participants,
                Function function,
                void* context
            ) {
                if (busy_.test_and_set(std::memory_order_acquire)) {
                    function(context, 0, 0, range);
                    return;
                }
                {
                    std::lock_guard lock(state_mutex_);
                    range_ = range;
                    unit_ = unit;
                    active_workers_ = participants - 1;
                    function_ = function;
                    context_ = context;
                    pending_ = active_workers_;
                    ++generation_;
                }
                for (std::size_t i = 0; i < active_workers_; ++i) {
                    ready_[i]->notify_one();
                }

                const auto wait_for_workers = [this] {
                    std::unique_lock lock(state_mutex_);
                    finished_.wait(lock, [this] {
                        return pending_ == 0;
                    });
                };

                try {
                    function(
                        context,
                        0,
                        0,
                        std::min(
                            partition_begin(range, participants, unit, 1),
                            range
                        )
                    );
                } catch (...) {
                    wait_for_workers();
                    busy_.clear(std::memory_order_release);
                    throw;
                }

                wait_for_workers();
                busy_.clear(std::memory_order_release);
            }

        private:
            void stop() {
                {
                    std::lock_guard lock(state_mutex_);
                    stopping_ = true;
                    ++generation_;
                }
                for (const auto& ready : ready_) {
                    ready->notify_one();
                }
                for (auto& worker : workers_) {
                    worker.join();
                }
            }

            void worker(std::size_t index) {
                std::size_t observed = 0;
                while (true) {
                    std::unique_lock lock(state_mutex_);
                    ready_[index - 1]->wait(lock, [this, observed] {
                        return stopping_ || generation_ != observed;
                    });
                    if (stopping_) {
                        return;
                    }

                    observed = generation_;
                    if (index > active_workers_) {
                        continue;
                    }
                    const auto participants = active_workers_ + 1;
                    const auto begin = std::min(
                        partition_begin(range_, participants, unit_, index),
                        range_
                    );
                    const auto end = std::min(
                        partition_begin(
                            range_, participants, unit_, index + 1
                        ),
                        range_
                    );
                    const auto function = function_;
                    auto* context = context_;
                    lock.unlock();

                    if (begin != end) {
                        function(context, index, begin, end);
                    }

                    lock.lock();
                    --pending_;
                    if (pending_ == 0) {
                        finished_.notify_one();
                    }
                }
            }

            std::vector<std::thread> workers_;
            std::vector<std::unique_ptr<std::condition_variable>> ready_;
            std::atomic_flag busy_ = ATOMIC_FLAG_INIT;
            std::mutex state_mutex_;
            std::condition_variable finished_;
            std::size_t generation_ = 0;
            std::size_t active_workers_ = 0;
            std::size_t pending_ = 0;
            std::size_t range_ = 0;
            std::size_t unit_ = 1;
            Function function_ = nullptr;
            void* context_ = nullptr;
            bool stopping_ = false;
        };

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
        const auto units = range / unit +
            static_cast<std::size_t>(range % unit != 0);
        const bool use_pool =
            work >= parallel_threshold &&
            units >= 2 &&
            thread_count() > 1;
        if (!use_pool) {
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
