#pragma once

#include <ormi/core/device.h>
#include <ormi/core/memory/allocator.h>

namespace ormi::backends::cuda {
    core::memory::Allocator* allocator(core::Device device);
}
