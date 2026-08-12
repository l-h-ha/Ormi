#include <ormi/core/device.h>
#include <ormi/core/error.h>
#include <ormi/backend/api.h>

#include "core/runtime/dispatch/device.h"
#include "core/runtime/device.h"

namespace ormi::core::runtime::device {
    memory::Allocator* allocator(Device device) {
        auto* result = backend(device).memory.allocator(device);

        if (!result) {
            ORMI_THROW(DeviceError, "Backend returned no allocator.");
        }

        return result;
    }

    void synchronize(Device device) {
        backend(device).memory.synchronize(device);
    }
}
