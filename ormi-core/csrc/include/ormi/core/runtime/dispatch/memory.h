#pragma once

#include <cstddef>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>

namespace ormi::core::runtime::memory {
    void copy(Device dst_device, void* dst, Device src_device, const void* src, std::size_t bytes);
    void fill(Device device, void* dst, unsigned char value, std::size_t bytes);
    void fill_scalar(Device device, void* dst, const void* value, std::size_t elements, DType dtype);
}
