#include <ormi/core/runtime/backend.h>
#include <ormi/core/runtime/dispatch/cast.h>
#include <ormi/core/tensor.h>

namespace ormi::core::runtime::cast {
    void run(Tensor& out, const Tensor& input) {
        out.device().backend()->cast(out, input);
    }
}
