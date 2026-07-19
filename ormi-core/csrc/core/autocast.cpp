#include <ormi/core/autocast.h>
#include <ormi/core/error.h>

#include "core/precision.h"

namespace ormi::core::precision {
    thread_local State state;
}

namespace ormi::core {
    Autocast::Autocast(DType dtype)
        : previous_enabled_(precision::state.enabled),
          previous_dtype_(precision::state.dtype)
    {
        if (dtype != DType::float16) {
            ORMI_THROW(DTypeError, "Autocast currently supports Float16 only.");
        }

        precision::state.enabled = true;
        precision::state.dtype = dtype;
    }

    Autocast::~Autocast() {
        precision::state.enabled = previous_enabled_;
        precision::state.dtype = previous_dtype_;
    }
}
