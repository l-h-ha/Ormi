#include <string>

#include <ormi/core/error.h>
#include <ormi/backend/api.h>

#include "core/runtime/dispatch/fractional.h"
#include "core/runtime/device.h"

namespace ormi::core::runtime::fractional {
    Setup prepare(
        ops::FractionalOp op,
        Tensor& out,
        const Caputo& attrs
    ) {
        const auto& selected = backend(out.device());
        if (!selected.fractional.prepare) {
            return {};
        }
        return selected.fractional.prepare(op, out, attrs);
    }

    void apply(
        ops::FractionalOp op,
        Tensor& out,
        const Tensor& input,
        const Tensor* initial_derivative,
        const Caputo& attrs
    ) {
        const auto& selected = backend(out.device());
        selected.fractional.execute(
            op,
            out,
            input,
            initial_derivative,
            attrs
        );
    }
}
