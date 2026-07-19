#include <ormi/core/runtime/backend.h>
#include <ormi/core/runtime/dispatch/pointwise.h>

namespace ormi::core::runtime::pointwise {
    void binary(ops::BinaryOp op, Tensor& out, const Tensor& lhs, const Tensor& rhs) {
        out.device().backend()->pointwise_binary(op, out, lhs, rhs);
    }

    void unary(ops::UnaryOp op, Tensor& out, const Tensor& t) {
        out.device().backend()->pointwise_unary(op, out, t);
    }
}
