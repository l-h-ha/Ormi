#pragma once
#include <cstddef>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/memory/buffer.h>

namespace ormi::core::memory {
    void copy(void* dst, Device dst_device, const void* src, Device src_device, std::size_t bytes);
    void copy(Buffer& dst, const Buffer& src);
    void copy(Buffer& dst, const Buffer& src, std::size_t bytes);

    // Repeat one byte, as memset does. Use fill_scalar for numerical values.
    void fill(void* dst, Device device, unsigned char value, std::size_t bytes);
    void fill(Buffer& dst, unsigned char value);
    // Repeat a host scalar of exactly dtype into elements values on device.
    void fill_scalar(void* dst, Device device, const void* value, std::size_t elements, DType dtype);

    void zero(void* dst, Device device, std::size_t bytes);
    void zero(Buffer& dst);
}
