#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "backends/cpu/executor.h"

namespace ormi::backends::cpu::executor {
    class Pool {
    public:
        explicit Pool(std::size_t thread_count);
        ~Pool();

        Pool(const Pool&) = delete;
        Pool& operator=(const Pool&) = delete;

        void run(
            std::size_t range,
            std::size_t unit,
            std::size_t participants,
            Function function,
            void* context
        );

    private:
        void stop();
        void worker(std::size_t index);

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
}
