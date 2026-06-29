#include <memory>
#include "core/buffer.h"
#include "optimizer.h"

namespace ormi::cpu {
    void sgd_step(const std::shared_ptr<Buffer>& grad, const std::shared_ptr<Buffer>& weight, float lr) {
        float* w_ptr = static_cast<float*>(weight->ptr);
        const float* g_ptr = static_cast<const float*>(grad->ptr);
        
        int elements = weight->bytes / sizeof(float);
        for (int i = 0; i < elements; ++i) {
            w_ptr[i] -= g_ptr[i] * lr;
        }
    }
}