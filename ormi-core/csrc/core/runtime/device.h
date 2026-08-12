#pragma once

#include <ormi/backend/api.h>
#include <ormi/core/device.h>

namespace ormi::core::runtime {
    inline const Backend& backend(const Device& device) {
        return *device.backend_;
    }
}
