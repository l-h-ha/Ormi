#include <stdexcept>
#include "allocator.h"
#include "core/error.h"

namespace ormi {
    IAllocator* get_allocator(Device dev) {
        if (dev.type == DeviceEnum::CPU) {
            return get_cpu_allocator();
        }

        #ifdef ORMI_USE_CUDA
        else if (dev.type == DeviceEnum::CUDA) {
            return get_cuda_allocator();
        }
        #endif

        else {
            ORMI_THROW(
                DeviceError,
                "Unrecognized DeviceEnum (" + to_string(dev) + ")"
            );
        }
    }
}