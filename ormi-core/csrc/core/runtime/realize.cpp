#include <ormi/core/tensor.h>

#include "core/runtime/realize.h"
#include "core/runtime/schedule.h"

namespace ormi::core::runtime {
    void realize(Tensor& tensor) {
        schedule::run(tensor);
    }
}
