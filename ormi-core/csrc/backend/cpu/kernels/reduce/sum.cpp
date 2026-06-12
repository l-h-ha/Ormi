#include <memory>
#include "core/memory.h"
#include "core/node.h"
#include "cpu_sum.h"
#include "backend/graph_breakers.h"

namespace ormi::cpu {
    std::shared_ptr<Buffer> sum(
        std::shared_ptr<Buffer> in_buf, 
        const TensorView& in_view,
        const std::vector<int>& axes,
        const TensorView& out_view,
        const bool keepdims
    ) {
        auto out_buffer = zeros(out_view.shape, DeviceEnum::CPU);

        float* in_ptr = static_cast<float*>(in_buf->ptr);
        float* out_ptr = static_cast<float*>(out_buffer->ptr);

        int in_elements = in_buf->bytes / sizeof(float);

        for (int i = 0; i < in_elements; ++i) {
            int remaining = i;
            int out_flat = 0;
            int out_d = 0;

            for (int d = 0; d < in_view.shape.size(); ++d) {
                int coord = remaining / in_view.strides[d];
                remaining %= in_view.strides[d];

                bool is_reduced = false;
                if (axes.empty()) {
                    is_reduced = true; 
                } else {
                    for (int ax : axes) {
                        if (ax == d) { is_reduced = true; break; }
                    }
                }
                
                if (!is_reduced) {
                    out_flat += coord * out_view.strides[out_d];
                    out_d++;
                } else if (keepdims) {
                    out_d++;
                }
            }
            out_ptr[out_flat] += in_ptr[i];
        }

        return out_buffer;
    }

    void sum_handler(std::shared_ptr<Node> root) {
        const auto& attrs = std::get<ReduceAttrs>(root->attrs);
        root->data = cpu::sum(
            root->parents[0]->data, 
            root->parents[0]->view, 
            attrs.axes, 
            root->view, 
            attrs.keepdims
        );
    }

    static RegisterGraphBreaker reg_cpu_sum(DeviceEnum::CPU, ops::OpEnum::SUM, sum_handler);
}