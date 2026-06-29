#include "utils.h"

namespace ormi::cpu {
    std::shared_ptr<Buffer> make_contiguous(const std::shared_ptr<Buffer>& in_buf, const TensorView& view) {
        if (view.is_contiguous()) return in_buf;

        auto out_buffer = empty(view.shape, Device::cpu(), in_buf->dtype);
        int elements = out_buffer->bytes / sizeof(float);
        
        const float* in_ptr = static_cast<const float*>(in_buf->ptr);
        float* out_ptr = static_cast<float*>(out_buffer->ptr);

        int ndim = view.shape.size();
        
        // unroll the memory
        for (int idx = 0; idx < elements; ++idx) {
            int offset = 0;
            int temp = idx;
            for (int i = ndim - 1; i >= 0; --i) {
                int coord = temp % view.shape[i];
                temp /= view.shape[i];
                offset += coord * view.strides[i];
            }
            out_ptr[idx] = in_ptr[offset];
        }

        return out_buffer;
    }
}