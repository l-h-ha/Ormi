#include <ormi/core/error.h>
#include <ormi/core/memory/buffer.h>
#include <ormi/backend/api.h>

#include "core/runtime/dispatch/memory.h"
#include "core/runtime/catalog.h"
#include "core/runtime/device.h"

namespace ormi::core::runtime::memory {
    void copy(Device dst_device, void* dst, Device src_device, const void* src, std::size_t bytes) {
        if (bytes == 0) {
            return;
        }

        if (!dst || !src) {
            ORMI_THROW(MemoryError, "Cannot copy memory with a null pointer.");
        }

        if (dst_device.backend_id() == src_device.backend_id()) {
            backend(dst_device).memory.copy(
                Transfer::within,
                dst_device,
                dst,
                src_device,
                src,
                bytes
            );
            return;
        }

        const bool dst_host = catalog::entries[dst_device.backend_id()].host;
        const bool src_host = catalog::entries[src_device.backend_id()].host;

        if (src_host) {
            backend(dst_device).memory.copy(
                Transfer::from_host,
                dst_device,
                dst,
                src_device,
                src,
                bytes
            );
            return;
        }

        if (dst_host) {
            backend(src_device).memory.copy(
                Transfer::to_host,
                dst_device,
                dst,
                src_device,
                src,
                bytes
            );
            return;
        }

        const auto& destination = backend(dst_device);
        if (
            destination.memory.direct_copy &&
            destination.memory.direct_copy(
                dst_device,
                dst,
                src_device,
                src,
                bytes
            )
        ) {
            return;
        }

        const Device host(catalog::host());
        core::memory::Buffer staging(host, bytes);

        backend(src_device).memory.copy(
            Transfer::to_host,
            host,
            staging.data(),
            src_device,
            src,
            bytes
        );
        destination.memory.copy(
            Transfer::from_host,
            dst_device,
            dst,
            host,
            staging.data(),
            bytes
        );
    }

    void fill(Device device, void* dst, unsigned char value, std::size_t bytes) {
        if (bytes == 0) {
            return;
        }

        if (!dst) {
            ORMI_THROW(MemoryError, "Cannot fill memory with a null pointer.");
        }

        backend(device).memory.fill(device, dst, value, bytes);
    }

    void fill_scalar(Device device, void* dst, const void* value, std::size_t elements, DType dtype) {
        if (elements == 0) {
            return;
        }

        if (!dst || !value) {
            ORMI_THROW(MemoryError, "Cannot fill tensor with a null pointer.");
        }

        backend(device).memory.fill_scalar(
            device,
            dst,
            value,
            elements,
            dtype
        );
    }
}
