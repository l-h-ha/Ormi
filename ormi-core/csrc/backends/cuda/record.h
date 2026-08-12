#pragma once

#include <ormi/core/device.h>
#include <ormi/backend/api.h>

namespace ormi::backends::cuda {
    void* record(
        core::Device device,
        core::runtime::RecordBodyFn body,
        void* context
    );
    void replay(void* recording);
    void release(void* recording) noexcept;
}
