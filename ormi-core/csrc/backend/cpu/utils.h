#pragma once
#include <memory>
#include "core/buffer.h"
#include "core/shape.h"

namespace ormi::cpu {
    std::shared_ptr<Buffer> make_contiguous(const std::shared_ptr<Buffer>& in_buf, const TensorView& view);
}