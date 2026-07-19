#include <ormi/core/device.h>
#include <ormi/core/error.h>
#include <ormi/core/runtime/backend.h>
#include <ormi/core/runtime/dispatch/device.h>

namespace ormi::core::runtime::device {
    memory::Allocator* allocator(Device device) {
        auto* result = device.backend()->allocator(device);

        if (!result) {
            ORMI_THROW(DeviceError, "Backend returned no allocator.");
        }

        return result;
    }

    void synchronize(Device device) {
        device.backend()->synchronize(device);
    }
}
