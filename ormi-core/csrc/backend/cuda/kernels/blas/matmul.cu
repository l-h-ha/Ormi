#include <cublas_v2.h>
#include <memory>
#include <vector>

#include "core/memory.h"
#include "core/node.h"
#include "cuda_matmul.cuh"
#include "backend/graph_breakers.h"

namespace ormi::cuda {
    cublasHandle_t get_cublas_handle() {
        static cublasHandle_t handle = nullptr;
        if (handle == nullptr) {
            cublasCreate(&handle);
        }
        return handle;
    }

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

        auto out_buffer = empty(out_view.shape, DeviceEnum::CUDA);

        const float* a_ptr = static_cast<const float*>(A->ptr);
        const float* b_ptr = static_cast<const float*>(B->ptr);
        float* c_ptr = static_cast<float*>(out_buffer->ptr);

        // Check if inner matrices are transposed using their strides
        cublasOperation_t trans_A = (view_A.strides[ndim_a - 1] == 1) ? CUBLAS_OP_N : CUBLAS_OP_T;
        cublasOperation_t trans_B = (view_B.strides[ndim_b - 1] == 1) ? CUBLAS_OP_N : CUBLAS_OP_T;

        // Leading dimensions based on transpose state
        int lda = (trans_A == CUBLAS_OP_N) ? view_A.strides[ndim_a - 2] : view_A.strides[ndim_a - 1];
        int ldb = (trans_B == CUBLAS_OP_N) ? view_B.strides[ndim_b - 2] : view_B.strides[ndim_b - 1];

        auto get_offset = [&](int flat_idx, const TensorView& target_view) {
            int offset = 0;
            int temp = flat_idx;
            for (int i = (int)out_batch_shape.size() - 1; i >= 0; --i) {
                int coord = temp % out_batch_shape[i];
                temp /= out_batch_shape[i];
                int target_dim_idx = i - (out_batch_shape.size() - (target_view.shape.size() - 2));
                if (target_dim_idx >= 0) {
                    if (target_view.shape[target_dim_idx] > 1) {
                        offset += coord * target_view.strides[target_dim_idx];
                    }
                }
            }
            return offset;
        };

        // Build the pointer arrays on the host
        std::vector<const float*> h_A_array(batch_count);
        std::vector<const float*> h_B_array(batch_count);
        std::vector<float*> h_C_array(batch_count);

        for (int b = 0; b < batch_count; ++b) {
            h_A_array[b] = a_ptr + get_offset(b, view_A);
            h_B_array[b] = b_ptr + get_offset(b, view_B);
            h_C_array[b] = c_ptr + (b * M * N);
        }

        const float** d_A_array;
        const float** d_B_array;
        float** d_C_array;
        
        cudaMalloc(&d_A_array, batch_count * sizeof(const float*));
        cudaMalloc(&d_B_array, batch_count * sizeof(const float*));
        cudaMalloc(&d_C_array, batch_count * sizeof(float*));

        cudaMemcpy(d_A_array, h_A_array.data(), batch_count * sizeof(const float*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_array, h_B_array.data(), batch_count * sizeof(const float*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_C_array, h_C_array.data(), batch_count * sizeof(float*), cudaMemcpyHostToDevice);

        cublasHandle_t handle = get_cublas_handle();
        float alpha = 1.0f;
        float beta = 0.0f;
        
        // B and A are swapped to handle row-major to col-major conversion
        cublasSgemmBatched(handle, trans_B, trans_A,
                           N, M, K,
                           &alpha,
                           d_B_array, ldb,
                           d_A_array, lda,
                           &beta,
                           d_C_array, N,
                           batch_count);

        cudaFree(d_A_array);
        cudaFree(d_B_array);
        cudaFree(d_C_array);

        return out_buffer;
    }

    void matmul_handler(std::shared_ptr<Node> root) {
        root->data = cuda::matmul(
            root->parents[0]->data, 
            root->parents[0]->view, 
            root->parents[1]->data, 
            root->parents[1]->view, 
            root->view
        );
    }

    static RegisterGraphBreaker reg_cuda_matmul(DeviceEnum::CUDA, ops::OpEnum::MATMUL, matmul_handler);
}