#include <ormi/backend/api.h>
#include <ormi/core/tensor.h>

#include "core/runtime/dispatch/optimizer.h"
#include "core/runtime/device.h"

namespace ormi::core::runtime::optimizer {
    void sgd(
        Tensor& out,
        const Tensor& parameter,
        const Tensor& gradient,
        double step_size
    ) {
        backend(out.device()).ops.sgd(
            out,
            parameter,
            gradient,
            step_size
        );
    }
}
