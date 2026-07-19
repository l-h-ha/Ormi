#include <ormi/core/error.h>
#include <ormi/core/runtime/backend.h>
#include <ormi/core/runtime/backend_id.h>
#include <ormi/core/runtime/dispatch/memory.h>

namespace ormi::core::runtime::memory {
    void copy(Device dst_device, void* dst, Device src_device, const void* src, std::size_t bytes) {
        if (bytes == 0) {
            return;
        }

        if (!dst || !src) {
            ORMI_THROW(MemoryError, "Cannot copy memory with a null pointer.");
        }

        const auto* backend = dst_device.backend_id() == src_device.backend_id()
            ? dst_device.backend()
            : (dst_device.backend_id() == backend_id_cpu ? src_device.backend() : dst_device.backend());

        backend->copy(dst_device, dst, src_device, src, bytes);
    }

    void fill(Device device, void* dst, unsigned char value, std::size_t bytes) {
        if (bytes == 0) {
            return;
        }

        if (!dst) {
            ORMI_THROW(MemoryError, "Cannot fill memory with a null pointer.");
        }

        device.backend()->fill(device, dst, value, bytes);
    }

    void fill_scalar(Device device, void* dst, const void* value, std::size_t elements, DType dtype) {
        if (elements == 0) {
            return;
        }

        if (!dst || !value) {
            ORMI_THROW(MemoryError, "Cannot fill tensor with a null pointer.");
        }

        device.backend()->fill_scalar(device, dst, value, elements, dtype);
    }
}
