#include <ormi/core/error.h>
#include <ormi/core/memory/ops.h>

#include "core/runtime/dispatch/memory.h"

namespace ormi::core::memory {
    void copy(void* dst, Device dst_device, const void* src, Device src_device, std::size_t bytes) {
        runtime::memory::copy(dst_device, dst, src_device, src, bytes);
    }

    void copy(Buffer& dst, const Buffer& src) {
        copy(dst, src, src.bytes());
    }

    void copy(Buffer& dst, const Buffer& src, std::size_t bytes) {
        if (bytes > dst.bytes() || bytes > src.bytes()) {
            ORMI_THROW(MemoryError, "Memory copy exceeds buffer size.");
        }

        copy(dst.data(), dst.device(), src.data(), src.device(), bytes);
    }

    void fill(void* dst, Device device, unsigned char value, std::size_t bytes) {
        runtime::memory::fill(device, dst, value, bytes);
    }

    void fill(Buffer& dst, unsigned char value) {
        fill(dst.data(), dst.device(), value, dst.bytes());
    }

    void fill_scalar(void* dst, Device device, const void* value, std::size_t elements, DType dtype) {
        runtime::memory::fill_scalar(device, dst, value, elements, dtype);
    }

    void zero(void* dst, Device device, std::size_t bytes) {
        fill(dst, device, 0, bytes);
    }

    void zero(Buffer& dst) {
        zero(dst.data(), dst.device(), dst.bytes());
    }
}
