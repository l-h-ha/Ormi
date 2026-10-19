#include <ormi/core/error.h>
#include <ormi/core/tensor.h>

#include "core/runtime/plan/impl.h"

namespace ormi::core {
    void Plan::run() {
        impl_->run();
    }

    void Plan::Impl::run() {
        for (auto& step : steps_) {
            step.run();
        }

        for (const auto& copy : copies_) {
            const auto bytes = copy.destination->bytes();
            if (bytes == 0) {
                continue;
            }

            auto* destination = copy.destination->data();
            const auto* value = copy.value->data();
            if (destination == value) {
                continue;
            }
            if (copy.destination->buffer_ == copy.value->buffer_) {
                ORMI_THROW(
                    OperationError,
                    "A plan update cannot copy between overlapping aliases."
                );
            }

            copy.execute(
                copy.destination->device_,
                destination,
                value,
                bytes
            );
        }

        ran_ = true;
    }

}
