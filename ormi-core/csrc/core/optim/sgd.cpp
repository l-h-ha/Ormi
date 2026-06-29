#include "sgd.h"

#include "backend/cpu/optimizer.h"
#include "core/error.h"

#ifdef ORMI_USE_CUDA
    #include "backend/cuda/optimizer.cuh"
#endif

namespace ormi::optim {
    SGD::SGD(
        const std::vector<std::pair<std::shared_ptr<Node>, 
        std::shared_ptr<Node>>>& parameter_grads, 
        float learning_rate
    ) : lr(learning_rate) {
        params.reserve(parameter_grads.size());

        for (const auto& pair : parameter_grads) {
            params.push_back({
                pair.first,
                pair.second
            });
        }
    }

    void SGD::step() {
        if (lr == 0.0f) {
            return;
        }

        for (auto& item : params) {
            const auto& param = item.param;
            const auto& grad = item.grad;

            if (!param || !param->data) {
                continue;
            }

            if (!grad || !grad->data) {
                continue;
            }

            if (param->device != grad->device) {
                ORMI_THROW(DeviceError, "SGD parameter and gradient device mismatch.");
            }

            if (param->dtype != grad->dtype) {
                ORMI_THROW(DTypeError, "SGD parameter and gradient dtype mismatch.");
            }

            if (param->data->bytes != grad->data->bytes) {
                ORMI_THROW(ShapeError, "SGD parameter and gradient size mismatch.");
            }

            if (param->device.type == DeviceEnum::CPU) {
                cpu::sgd_step(grad->data, param->data, lr);
            }

    #ifdef ORMI_USE_CUDA
            else if (param->device.type == DeviceEnum::CUDA) {
                cuda::sgd_step(grad->data, param->data, lr);
            }
    #endif

            else {
                ORMI_THROW(DeviceError, "Unsupported device for SGD.");
            }
        }
    }
}
