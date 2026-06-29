#include "zero_grad_tape.h"

#include "core/allocator.h"
#include "core/error.h"
#include "core/ops.h"

namespace ormi {
    void ZeroGradTape::capture(const std::vector<std::shared_ptr<Node>>& grads) {
        blocks.clear();

        for (const auto& grad : grads) {
            if (!grad || !grad->data || !grad->data->ptr) {
                continue;
            }

            blocks.push_back({
                grad->data->ptr,
                grad->data->bytes,
                grad->data->device
            });
        }
    }

    void ZeroGradTape::execute() {
        for (const auto& block : blocks) {
            if (!block.ptr || block.bytes == 0) {
                continue;
            }

            get_allocator(block.device)->memset(
                block.ptr,
                0,
                block.bytes
            );
        }
    }

}