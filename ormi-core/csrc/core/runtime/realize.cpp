#include <memory>

#include <ormi/core/memory/buffer.h>
#include <ormi/core/runtime/backend.h>
#include <ormi/core/runtime/realize.h>
#include <ormi/core/tensor.h>

#include "core/runtime/fusion/expr.h"

namespace ormi::core::runtime {
    void realize(Tensor& tensor) {
        if (tensor.buffer_) {
            return;
        }

        const auto* backend = tensor.device_.backend();

        if (backend->fusion) {
            const auto expression = fusion::build(tensor);

            if (expression) {
                for (auto* input : expression->inputs) {
                    input->realize();
                }

                tensor.buffer_ = std::make_shared<memory::Buffer>(tensor.device_, tensor.bytes());

                try {
                    backend->fusion(tensor, *expression);
                } catch (...) {
                    tensor.buffer_.reset();
                    throw;
                }
                return;
            }
        }

        for (const auto& parent : tensor.parents_) {
            parent->realize();
        }

        if (tensor.op_->output == Output::alias) {
            tensor.buffer_ = tensor.parents_[0]->buffer_;
            return;
        }

        tensor.buffer_ = std::make_shared<memory::Buffer>(tensor.device_, tensor.bytes());

        try {
            tensor.op_->compute(tensor);
        } catch (...) {
            tensor.buffer_.reset();
            throw;
        }
    }
}
