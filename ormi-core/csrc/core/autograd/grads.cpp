#include <memory>

#include <ormi/core/autograd.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/tensor.h>

namespace ormi::core {
    void Grads::add(const std::shared_ptr<Tensor>& tensor, std::shared_ptr<Tensor> gradient) {
        if (!tensor || !gradient) {
            ORMI_THROW(OperationError, "Gradient accumulation requires a tensor and a gradient.");
        }

        if (tensor->shape() != gradient->shape()) {
            ORMI_THROW(ShapeError, "Accumulated gradient shape must match its tensor.");
        }

        if (tensor->dtype() != gradient->dtype()) {
            ORMI_THROW(DTypeError, "Accumulated gradient dtype must match its tensor.");
        }

        const auto tensor_device = tensor->device();
        const auto gradient_device = gradient->device();

        if (
            tensor_device.backend_id() != gradient_device.backend_id() ||
            tensor_device.index() != gradient_device.index()
        ) {
            ORMI_THROW(DeviceError, "Accumulated gradient device must match its tensor.");
        }

        const auto found = values_.find(tensor.get());

        if (found == values_.end()) {
            values_.emplace(tensor.get(), std::move(gradient));
        } else {
            found->second = ops::add(found->second, gradient);
        }
    }

    std::shared_ptr<Tensor> Grads::get(const std::shared_ptr<Tensor>& tensor) const {
        if (!tensor) {
            return nullptr;
        }

        const auto found = values_.find(tensor.get());
        return found == values_.end() ? nullptr : found->second;
    }
}
