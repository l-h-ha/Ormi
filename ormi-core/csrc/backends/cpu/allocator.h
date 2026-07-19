#pragma once

#include <ormi/core/device.h>
#include <ormi/core/memory/allocator.h>

namespace ormi::backends::cpu {
    core::memory::Allocator* allocator();
    core::memory::Allocator* get_allocator(core::Device device);
}
