#include <array>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <future>
#include <initializer_list>
#include <stdexcept>
#include <string_view>

#include <ormi/core/error.h>

#include "backends/cpu/executor.h"

namespace {
    namespace executor = ormi::backends::cpu::executor;

    void set_threads(const char* value) {
    #ifdef _WIN32
        _putenv_s("ORMI_CPU_THREADS", value);
    #else
        setenv("ORMI_CPU_THREADS", value, 1);
    #endif
    }

    struct Call {
        std::size_t count = 0;
        std::size_t begin = 1;
        std::size_t end = 1;
    };

    void count(void* context, std::size_t, std::size_t begin, std::size_t end) {
        auto& call = *static_cast<Call*>(context);
        ++call.count;
        call.begin = begin;
        call.end = end;
    }

    bool rejects_configuration() {
        try {
            executor::slots(1024, 1, 1024 * 1024);
        } catch (const ormi::OperationError& error) {
            return std::string_view(error.what()) ==
                "ORMI_CPU_THREADS must be a positive integer.";
        }
        return false;
    }

    bool lazy_configuration() {
        set_threads("invalid");
        if (executor::slots(1024, 1, 0) != 1 ||
            executor::slots(0, 1, 1024 * 1024) != 1 ||
            executor::slots(1, 1, 1024 * 1024) != 1) {
            return false;
        }

        Call empty;
        executor::run(0, 1, 1024 * 1024, count, &empty);
        if (empty.count != 1 || empty.begin != 0 || empty.end != 0) {
            return false;
        }
        try {
            executor::slots(1024, 0, 1024 * 1024);
            return false;
        } catch (const ormi::OperationError& error) {
            if (std::string_view(error.what()) !=
                "CPU execution partition unit must be positive.") {
                return false;
            }
        }
        for (const auto* value : {"invalid", "0", "-1", "4x", " 4", "99999999999999999999999"}) {
            set_threads(value);
            if (!rejects_configuration()) {
                return false;
            }
        }

        set_threads("4");
        if (executor::slots(1024, 1, 512 * 1024) != 3 ||
            executor::slots(1024, 1, 1024 * 1024) != 4) {
            return false;
        }
        set_threads("invalid");
        return executor::slots(1024, 1, 1024 * 1024) == 4;
    }

    void throw_on_worker(void*, std::size_t slot, std::size_t, std::size_t) {
        if (slot == 1) {
            std::set_terminate([]() noexcept { std::_Exit(73); });
            throw std::runtime_error("worker callback");
        }
    }

    struct Visits {
        std::array<std::atomic<unsigned>, 67> values{};
        std::array<std::atomic<unsigned>, 4> calls{};
        std::atomic<bool> valid{true};
    };

    void visit(void* context, std::size_t slot, std::size_t begin, std::size_t end) {
        auto& visits = *static_cast<Visits*>(context);
        if (slot >= visits.calls.size() || begin >= end ||
            end > visits.values.size() || begin % 4 != 0 ||
            (end != visits.values.size() && end % 4 != 0)) {
            visits.valid = false;
            return;
        }
        ++visits.calls[slot];
        for (auto index = begin; index < end; ++index) {
            ++visits.values[index];
        }
    }

    bool complete(const Visits& visits) {
        if (!visits.valid || visits.calls[0] != 1) {
            return false;
        }
        const auto worker_calls = visits.calls[1].load();
        if (worker_calls > 1 || visits.calls[2] != worker_calls ||
            visits.calls[3] != worker_calls) {
            return false;
        }
        for (const auto& count : visits.values) {
            if (count != 1) {
                return false;
            }
        }
        return true;
    }

    bool concurrent_startup() {
        set_threads("4");
        std::promise<void> start;
        const auto ready = start.get_future().share();
        std::array<std::future<bool>, 8> submitters;
        for (auto& submitter : submitters) {
            submitter = std::async(std::launch::async, [ready] {
                ready.wait();
                for (unsigned iteration = 0; iteration < 32; ++iteration) {
                    Visits visits;
                    executor::run(67, 4, 1024 * 1024, visit, &visits);
                    if (!complete(visits)) {
                        return false;
                    }
                }
                return true;
            });
        }
        start.set_value();
        bool valid = true;
        for (auto& submitter : submitters) {
            if (!submitter.get()) {
                valid = false;
            }
        }

        Visits final;
        executor::run(67, 4, 1024 * 1024, visit, &final);
        return valid && complete(final) && final.calls[1] == 1;
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        return 1;
    }
    const std::string_view mode = argv[1];
    if (mode == "lazy") {
        return lazy_configuration() ? 0 : 1;
    }
    if (mode == "single") {
        set_threads("1");
        Call call;
        executor::run(1024, 1, 1024 * 1024, count, &call);
        return call.count == 1 && call.begin == 0 && call.end == 1024 ? 0 : 1;
    }
    if (mode == "worker-throws") {
        set_threads("4");
        executor::run(1024, 1, 1024 * 1024, throw_on_worker, nullptr);
    }
    if (mode == "startup") {
        return concurrent_startup() ? 0 : 1;
    }
    return 1;
}
