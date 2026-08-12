#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/backend/api.h>
#include <ormi/core/tensor.h>

#include "core/runtime/dispatch/finite.h"
#include "core/runtime/device.h"

namespace ormi::core::runtime::finite {
    bool all(std::span<const Tensor* const> inputs) {
        if (inputs.empty()) {
            ORMI_THROW(OperationError, "Finiteness query requires at least one tensor.");
        }

        const auto* first = inputs.front();
        if (first == nullptr) {
            ORMI_THROW(OperationError, "Finiteness query received a null tensor.");
        }

        const auto device = first->device();

        for (const auto* input : inputs) {
            if (input == nullptr) {
                ORMI_THROW(OperationError, "Finiteness query received a null tensor.");
            }

            if (!is_floating(input->dtype())) {
                ORMI_THROW(DTypeError, "Finiteness query supports floating-point tensors only.");
            }

            if (
                input->device().backend_id() != device.backend_id() ||
                input->device().index() != device.index()
            ) {
                ORMI_THROW(DeviceError, "Finiteness query tensors must use one device.");
            }
        }

        const auto& selected = backend(device);
        return selected.ops.all_finite(inputs);
    }
}
