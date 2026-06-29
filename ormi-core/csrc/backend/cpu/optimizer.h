#pragma once
#include <memory>
#include "core/buffer.h"

namespace ormi::cpu {
    void sgd_step(const std::shared_ptr<Buffer>& grad, const std::shared_ptr<Buffer>& weight, float lr);
}
