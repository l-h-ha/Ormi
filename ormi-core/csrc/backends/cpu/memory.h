#pragma once
#include <cstddef>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/backend/api.h>

namespace ormi::backends::cpu {
    void copy(
        core::runtime::Transfer transfer,
        core::Device dst_device,
        void* dst,
        core::Device src_device,
        const void* src,
        std::size_t bytes
    );
    void copy_async(
        core::Device device,
        void* dst,
        const void* src,
        std::size_t bytes
    );
    void fill(core::Device device, void* dst, unsigned char value, std::size_t bytes);
    void fill_scalar(
        core::Device device,
        void* dst,
        const void* value,
        std::size_t elements,
        core::DType dtype
    );
}
