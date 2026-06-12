#include <stdexcept>
#include "allocator.h"

namespace ormi {
    IAllocator* get_allocator(DeviceEnum type) {
        if (type == DeviceEnum::CPU) {
            return get_cpu_allocator();
        } else if (type == DeviceEnum::CUDA) {
            return get_cuda_allocator();
        }
        throw std::runtime_error("Unsupported DeviceEnum.");
    }
}