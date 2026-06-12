#include <memory>
#include <vector>
#include <cblas.h>
#include "core/memory.h"
#include "core/node.h"
#include "cpu_matmul.h"
#include "backend/graph_breakers.h"

namespace ormi::cpu {
    std::shared_ptr<Buffer> matmul(
        std::shared_ptr<Buffer> A, 
        const TensorView& view_A,
        std::shared_ptr<Buffer> B, 
        const TensorView& view_B,
        const TensorView& out_view
    ) {
        int ndim_a = view_A.shape.size();
        int ndim_b = view_B.shape.size();

        int M = view_A.shape[ndim_a - 2];
        int K = view_A.shape[ndim_a - 1];
        int N = view_B.shape[ndim_b - 1];

        int batch_count = 1;
        std::vector<int> out_batch_shape;
        for (int i = 0; i < (int)out_view.shape.size() - 2; ++i) {
            batch_count *= out_view.shape[i];
            out_batch_shape.push_back(out_view.shape[i]);
        }

        auto out_buffer = empty(out_view.shape, DeviceEnum::CPU);

        const float* a_ptr = static_cast<const float*>(A->ptr);
        const float* b_ptr = static_cast<const float*>(B->ptr);
        float* c_ptr = static_cast<float*>(out_buffer->ptr);
        
        CBLAS_TRANSPOSE trans_A = (view_A.strides[ndim_a - 1] == 1) ? CblasNoTrans : CblasTrans;
        CBLAS_TRANSPOSE trans_B = (view_B.strides[ndim_b - 1] == 1) ? CblasNoTrans : CblasTrans;

        int lda = (trans_A == CblasNoTrans) ? view_A.strides[ndim_a - 2] : view_A.strides[ndim_a - 1];
        int ldb = (trans_B == CblasNoTrans) ? view_B.strides[ndim_b - 2] : view_B.strides[ndim_b - 1];

        auto get_offset = [&](int flat_idx, const TensorView& target_view) {
            int offset = 0;
            int temp = flat_idx;
            
            for (int i = (int)out_batch_shape.size() - 1; i >= 0; --i) {
                int coord = temp % out_batch_shape[i];
                temp /= out_batch_shape[i];
                
                int target_dim_idx = i - (out_batch_shape.size() - (target_view.shape.size() - 2));
                if (target_dim_idx >= 0) {
                    if (target_view.shape[target_dim_idx] > 1) {
                        // Use the explicitly provided stride!
                        offset += coord * target_view.strides[target_dim_idx];
                    }
                }
            }
            return offset;
        };

        for (int b = 0; b < batch_count; ++b) {
            int offset_a = get_offset(b, view_A);
            int offset_b = get_offset(b, view_B);

            const float* a_batch = a_ptr + offset_a;
            const float* b_batch = b_ptr + offset_b;
            float* c_batch = c_ptr + (b * M * N);

            cblas_sgemm(CblasRowMajor, trans_A, trans_B,
                        M, N, K, 1.0f,
                        a_batch, lda, b_batch, ldb, 0.0f,
                        c_batch, N);
        }

        return out_buffer;
    }

    void matmul_handler(std::shared_ptr<Node> root) {
        root->data = cpu::matmul(
            root->parents[0]->data, 
            root->parents[0]->view, 
            root->parents[1]->data, 
            root->parents[1]->view, 
            root->view
        );
    }

    static RegisterGraphBreaker reg_cpu_matmul(DeviceEnum::CPU, ops::OpEnum::MATMUL, matmul_handler);
}