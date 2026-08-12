#include <ormi/backend/api.h>
#include <ormi/core/tensor.h>

#include "core/runtime/dispatch/cast.h"
#include "core/runtime/device.h"

namespace ormi::core::runtime::cast {
    void run(Tensor& out, const Tensor& input) {
        backend(out.device()).ops.cast(out, input);
    }
}
