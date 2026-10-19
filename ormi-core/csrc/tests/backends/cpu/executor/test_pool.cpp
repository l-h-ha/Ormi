#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "backends/cpu/executor.h"

namespace {
    namespace executor = ormi::backends::cpu::executor;
    using namespace std::chrono_literals;
    constexpr std::size_t range = 67;
    constexpr std::size_t work = 1024 * 1024;

    struct CallbackError : std::runtime_error {
        CallbackError() : std::runtime_error("caller callback") {}
    };

    struct Blocked {
        std::mutex mutex;
        std::condition_variable changed;
        std::size_t entered = 0;
        std::size_t finished = 0;
        bool released = false;
        bool throw_on_caller = false;

        bool wait(std::size_t workers) {
            std::unique_lock lock(mutex);
            return changed.wait_for(lock, 5s, [&] { return entered == workers; });
        }

        void release() {
            std::lock_guard lock(mutex);
            released = true;
            changed.notify_all();
        }
    };

    void block_workers(void* context, std::size_t slot, std::size_t, std::size_t) {
        auto& blocked = *static_cast<Blocked*>(context);
        if (slot == 0) {
            if (blocked.throw_on_caller) {
                throw CallbackError();
            }
            return;
        }
        std::unique_lock lock(blocked.mutex);
        ++blocked.entered;
        blocked.changed.notify_all();
        blocked.changed.wait(lock, [&] { return blocked.released; });
        ++blocked.finished;
    }

    struct Visits {
        std::vector<std::atomic<unsigned>> values = std::vector<std::atomic<unsigned>>(range);
        std::atomic<unsigned> calls{0};
        std::atomic<bool> serial{true};
        std::atomic<bool> aligned{true};
    };

    void visit(void* context, std::size_t slot, std::size_t begin, std::size_t end) {
        auto& visits = *static_cast<Visits*>(context);
        ++visits.calls;
        if (slot != 0 || begin != 0 || end != range) {
            visits.serial = false;
        }
        if (begin % 4 != 0 || (end != range && end % 4 != 0)) {
            visits.aligned = false;
        }
        for (auto index = begin; index < end; ++index) {
            ++visits.values[index];
        }
    }

    bool visited_once(const Visits& visits) {
        if (!visits.aligned) {
            return false;
        }
        for (const auto& count : visits.values) {
            if (count != 1) {
                return false;
            }
        }
        return true;
    }

    void throw_on_caller(void*, std::size_t slot, std::size_t, std::size_t) {
        if (slot == 0) {
            throw CallbackError();
        }
    }

    void nested(void* context, std::size_t slot, std::size_t, std::size_t) {
        auto* visits = static_cast<Visits*>(context);
        executor::run(range, 4, work, visit, &visits[slot]);
    }

    void nested_throw(void* context, std::size_t, std::size_t, std::size_t) {
        auto& caught = *static_cast<std::atomic<unsigned>*>(context);
        try {
            executor::run(range, 4, work, throw_on_caller, nullptr);
        } catch (const CallbackError&) {
            ++caught;
        }
    }

    TEST(CPUExecutorPoolTest, waitsForWorkersBeforeReturning) {
        Blocked blocked;
        auto submission = std::async(std::launch::async, [&] {
            executor::run(range, 4, work, block_workers, &blocked);
        });
        const bool entered = blocked.wait(3);
        const auto status = submission.wait_for(50ms);
        blocked.release();
        submission.get();
        EXPECT_TRUE(entered);
        EXPECT_EQ(status, std::future_status::timeout);
        EXPECT_EQ(blocked.finished, 3u);
    }

    TEST(CPUExecutorPoolTest, runsContendedSubmissionSeriallyWithoutWaiting) {
        Blocked blocked;
        auto owner = std::async(std::launch::async, [&] {
            executor::run(range, 4, work, block_workers, &blocked);
        });
        const bool entered = blocked.wait(3);
        Visits visits;
        auto contender = std::async(std::launch::async, [&] {
            executor::run(range, 4, work, visit, &visits);
        });
        const auto status = contender.wait_for(1s);
        blocked.release();
        contender.get();
        owner.get();
        EXPECT_TRUE(entered);
        EXPECT_EQ(status, std::future_status::ready);
        EXPECT_EQ(visits.calls, 1u);
        EXPECT_TRUE(visits.serial);
        EXPECT_TRUE(visited_once(visits));
    }

    TEST(CPUExecutorPoolTest, contendedExceptionDoesNotReleaseAnotherSubmission) {
        Blocked blocked;
        auto owner = std::async(std::launch::async, [&] {
            executor::run(range, 4, work, block_workers, &blocked);
        });
        const bool entered = blocked.wait(3);
        auto throwing = std::async(std::launch::async, [] {
            try {
                executor::run(range, 4, work, throw_on_caller, nullptr);
            } catch (const CallbackError&) {
                return true;
            }
            return false;
        });
        const auto throwing_status = throwing.wait_for(1s);
        Visits visits;
        auto contender = std::async(std::launch::async, [&] {
            executor::run(range, 4, work, visit, &visits);
        });
        const auto status = contender.wait_for(1s);
        blocked.release();
        const bool caught = throwing.get();
        contender.get();
        owner.get();
        EXPECT_TRUE(entered);
        EXPECT_TRUE(caught);
        EXPECT_EQ(throwing_status, std::future_status::ready);
        EXPECT_EQ(status, std::future_status::ready);
        EXPECT_EQ(visits.calls, 1u);
        EXPECT_TRUE(visits.serial);
        EXPECT_TRUE(visited_once(visits));
    }

    TEST(CPUExecutorPoolTest, waitsBeforePropagatingCallerExceptionAndCanBeReused) {
        Blocked blocked;
        blocked.throw_on_caller = true;
        auto submission = std::async(std::launch::async, [&] {
            try {
                executor::run(range, 4, work, block_workers, &blocked);
            } catch (const CallbackError&) {
                return true;
            }
            return false;
        });
        const bool entered = blocked.wait(3);
        const auto status = submission.wait_for(50ms);
        blocked.release();
        const bool caught = submission.get();
        EXPECT_TRUE(entered);
        EXPECT_EQ(status, std::future_status::timeout);
        EXPECT_TRUE(caught);
        EXPECT_EQ(blocked.finished, 3u);

        Visits visits;
        executor::run(range, 4, work, visit, &visits);
        EXPECT_EQ(visits.calls, 4u);
        EXPECT_TRUE(visited_once(visits));
    }

    TEST(CPUExecutorPoolTest, runsNestedSubmissionsFromEveryParticipantSerially) {
        Visits visits[4];
        executor::run(range, 4, work, nested, visits);
        for (const auto& submission : visits) {
            EXPECT_EQ(submission.calls, 1u);
            EXPECT_TRUE(submission.serial);
            EXPECT_TRUE(visited_once(submission));
        }
    }

    TEST(CPUExecutorPoolTest, canCatchNestedExceptionsInsideWorkerCallbacks) {
        std::atomic<unsigned> caught{0};
        executor::run(range, 4, work, nested_throw, &caught);
        EXPECT_EQ(caught, 4u);
        Visits visits;
        executor::run(range, 4, work, visit, &visits);
        EXPECT_EQ(visits.calls, 4u);
        EXPECT_TRUE(visited_once(visits));
    }

    TEST(CPUExecutorPoolTest, changesParticipantCountWithoutReusingOldContext) {
        for (unsigned generation = 0; generation < 64; ++generation) {
            const std::size_t size = generation % 3 == 0 ? 8 : range;
            const std::size_t cost = generation % 3 == 1 ? 512 * 1024 : work;
            Visits visits;
            executor::run(size, 4, cost, visit, &visits);
            EXPECT_EQ(visits.calls, executor::slots(size, 4, cost));
            for (std::size_t index = 0; index < range; ++index) {
                EXPECT_EQ(visits.values[index], index < size ? 1u : 0u);
            }
        }
    }

    TEST(CPUExecutorPoolTest, growsParticipationAfterSmallFirstSubmission) {
        Visits small;
        executor::run(range, 4, 512 * 1024, visit, &small);
        EXPECT_EQ(small.calls, 3u);
        Visits large;
        executor::run(range, 4, work, visit, &large);
        EXPECT_EQ(large.calls, 4u);
        EXPECT_TRUE(visited_once(small));
        EXPECT_TRUE(visited_once(large));
    }

    TEST(CPUExecutorPoolTest, preservesContextAndCoverageWithConcurrentSubmitters) {
        std::promise<void> start;
        const auto ready = start.get_future().share();
        std::vector<std::future<bool>> submitters;
        for (unsigned thread = 0; thread < 8; ++thread) {
            submitters.push_back(std::async(std::launch::async, [ready] {
                ready.wait();
                for (unsigned iteration = 0; iteration < 64; ++iteration) {
                    Visits visits;
                    executor::run(range, 4, work, visit, &visits);
                    if (!visited_once(visits)) {
                        return false;
                    }
                }
                return true;
            }));
        }
        start.set_value();
        for (auto& submitter : submitters) {
            EXPECT_TRUE(submitter.get());
        }
    }
}
