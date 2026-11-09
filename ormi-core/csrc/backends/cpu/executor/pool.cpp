#include <algorithm>

#include "backends/cpu/executor/pool.h"

namespace ormi::backends::cpu::executor {
    namespace {
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
    }

    Pool::Pool(std::size_t thread_count) {
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

    Pool::~Pool() {
        stop();
    }

    void Pool::run(
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

    void Pool::stop() {
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

    void Pool::worker(std::size_t index) {
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
}
