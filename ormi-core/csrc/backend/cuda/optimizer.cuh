#pragma once
#include <cuda.h>
#include <memory>
#include "core/buffer.h"

namespace ormi::cuda {
    void sgd_step(const std::shared_ptr<Buffer>& grad, const std::shared_ptr<Buffer>& weight, float lr);
}