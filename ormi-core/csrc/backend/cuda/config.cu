#include <string>
#include "backend/cuda/config.cuh"
#include "core/error.h"

namespace ormi::cuda {
    namespace {
        CudaConfig global_config{};

        int validate_threads(int threads, const char* name) {
            if (threads <= 0) {
                ORMI_THROW(ValueError, std::string(name) + " must be positive.");
            }

            if (threads > 1024) {
                ORMI_THROW(ValueError, std::string(name) + " must be <= 1024.");
            }

            if (threads % 32 != 0) {
                ORMI_THROW(ValueError, std::string(name) + " must be a multiple of 32.");
            }

            return threads;
        }

        bool is_power_of_two(int x) {
            return x > 0 && (x & (x - 1)) == 0;
        }
    }

    CudaConfig get_config() {
        return global_config;
    }

    void set_config(const CudaConfig& config) {
        CudaConfig checked = config;

        checked.optimizer_threads = validate_threads(
            checked.optimizer_threads,
            "optimizer_threads"
        );

        checked.elementwise_threads = validate_threads(
            checked.elementwise_threads,
            "elementwise_threads"
        );

        checked.reduction_threads = validate_threads(
            checked.reduction_threads,
            "reduction_threads"
        );

        if (!is_power_of_two(checked.reduction_threads)) {
            ORMI_THROW(
                ValueError,
                "reduction_threads must be a power of two."
            );
        }

        global_config = checked;
    }

    int get_optimizer_threads() {
        return global_config.optimizer_threads;
    }

    int get_elementwise_threads() {
        return global_config.elementwise_threads;
    }

    int get_reduction_threads() {
        return global_config.reduction_threads;
    }

    std::ostream& operator<<(std::ostream& os, const CudaConfig& config) {
        os << "CudaConfig{"
        << "optimizer_threads=" << config.optimizer_threads
        << ", elementwise_threads=" << config.elementwise_threads
        << ", reduction_threads=" << config.reduction_threads
        << "}";

        return os;
    }
}