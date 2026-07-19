#pragma once

#include <ormi/core/device.h>

namespace ormi::core::memory {
    class Allocator;
}

namespace ormi::core::runtime::device {
    memory::Allocator* allocator(Device device);
    void synchronize(Device device);
}
