#include <ormi/backend/api.h>

#include "core/runtime/dispatch/linalg.h"
#include "core/runtime/device.h"

namespace ormi::core::runtime::linalg {
    void matmul(Tensor& out, const Tensor& lhs, const Tensor& rhs) {
        backend(out.device()).ops.matmul(out, lhs, rhs);
    }
}
