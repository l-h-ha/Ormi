#include <ormi/backend/api.h>

#include "core/runtime/dispatch/pointwise.h"
#include "core/runtime/device.h"

namespace ormi::core::runtime::pointwise {
    void binary(ops::BinaryOp op, Tensor& out, const Tensor& lhs, const Tensor& rhs) {
        backend(out.device()).ops.binary(op, out, lhs, rhs);
    }

    void unary(ops::UnaryOp op, Tensor& out, const Tensor& t) {
        backend(out.device()).ops.unary(op, out, t);
    }
}
