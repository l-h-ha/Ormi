#pragma once

#include <cuda_runtime_api.h>

#include <ormi/core/error.h>

#include "check.cuh"

namespace ormi::backends::cuda {
    inline int available_devices() {
        int count = 0;
        check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
        return count;
    }

    inline void check_device_id(int id) {
        const int count = available_devices();

        if (id < 0 || id >= count) {
            ORMI_THROW(
                BackendError,
                "CUDA device index: " + std::to_string(id) + " is out of range. Only " +
                std::to_string(count) + " device(s) are available."
            );
        }
    }
}
