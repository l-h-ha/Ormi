#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/runtime/backend.h>
#include <ormi/core/runtime/dispatch/finite.h>
#include <ormi/core/tensor.h>

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

        if (device.backend()->all_finite == nullptr) {
            ORMI_THROW(OperationError, "The selected backend does not support finiteness queries.");
        }

        return device.backend()->all_finite(inputs);
    }
}
