#pragma once
#include <cstddef>

namespace ormi::cuda {
    void ORMI_cudaMemsetAsync(void* ptr, int value, size_t bytes);
}