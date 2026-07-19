#include <ormi/core/runtime/backend.h>
#include <ormi/core/runtime/dispatch/linalg.h>

namespace ormi::core::runtime::linalg {
    void matmul(Tensor& out, const Tensor& lhs, const Tensor& rhs) {
        out.device().backend()->matmul(out, lhs, rhs);
    }
}
