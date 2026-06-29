#include <cuda.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <memory>
#include <vector>
#include <unordered_map>

#include "core/buffer.h"
#include "core/node.h"
#include "matmul.cuh"
#include "backend/graph_breakers.h"
#include "core/shape.h"
#include "utils/macros.h"
#include "backend/cuda/utils.cuh"
#include "backend/cuda/config.cuh"

namespace ormi::cuda {
    template <typename T>
    __global__ void copy_2d_batch(const T* in, T* out, int M, int K, int stride_m, int stride_k) {
        int r = blockIdx.y * blockDim.y + threadIdx.y;
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (r < M && c < K) {
            out[r * K + c] = in[r * stride_m + c * stride_k];
        }
    }

    namespace {
        dim3 make_copy2d_block() {
            const int total_threads = get_elementwise_threads();
            if (total_threads < 32 || total_threads % 32 != 0) {
                ORMI_THROW(
                    ValueError,
                    "elementwise_threads must be >= 32 and divisible by 32."
                );
            }

            // Keep x wide for coalesced column reads/writes.
            // 128 -> 32 x 4
            // 256 -> 32 x 8
            // 512 -> 32 x 16
            // 1024 -> 32 x 32
            const int block_x = 32;
            const int block_y = total_threads / block_x;

            return dim3(block_x, block_y);
        }

        dim3 make_copy2d_grid(int cols, int rows, dim3 block) {
            return dim3(
                (cols + static_cast<int>(block.x) - 1) / static_cast<int>(block.x),
                (rows + static_cast<int>(block.y) - 1) / static_cast<int>(block.y)
            );
        }
    }

    std::shared_ptr<Buffer> matmul(
        const std::shared_ptr<Buffer>& A,
        const TensorView& view_A,
        const std::shared_ptr<Buffer>& B,
        const TensorView& view_B,
        const TensorView& out_view,
        std::shared_ptr<Buffer> out_buffer
    ) {
        int ndim_a = view_A.shape.size();
        int ndim_b = view_B.shape.size();

        if (ndim_a < 2 || ndim_b < 2) ORMI_THROW(OperationError, "MATMUL requires at least 2D tensors.");
        if (A->dtype != B->dtype) ORMI_THROW(DTypeError, "DTypes must match.");

        int M = view_A.shape[ndim_a - 2];
        int K = view_A.shape[ndim_a - 1];
        int N = view_B.shape[ndim_b - 1];

        int batch_count = 1;
        std::vector<int> out_batch_shape;
        for (int i = 0; i < (int)out_view.shape.size() - 2; ++i) {
            batch_count *= out_view.shape[i];
            out_batch_shape.push_back(out_view.shape[i]);
        }

        if (!out_buffer) out_buffer = empty(out_view.shape, A->device, A->dtype);

        const char* a_ptr_raw = static_cast<const char*>(A->ptr);
        const char* b_ptr_raw = static_cast<const char*>(B->ptr);
        char* c_ptr_raw = static_cast<char*>(out_buffer->ptr);

        bool a_compat = false; int trans_A = CUBLAS_OP_N; int lda = K;
        if (view_A.strides[ndim_a - 1] == 1 && view_A.strides[ndim_a - 2] >= K) {
            a_compat = true; trans_A = CUBLAS_OP_N; lda = view_A.strides[ndim_a - 2];
        } else if (view_A.strides[ndim_a - 2] == 1 && view_A.strides[ndim_a - 1] >= M) {
            a_compat = true; trans_A = CUBLAS_OP_T; lda = view_A.strides[ndim_a - 1];
        }

        bool b_compat = false; int trans_B = CUBLAS_OP_N; int ldb = N;
        if (view_B.strides[ndim_b - 1] == 1 && view_B.strides[ndim_b - 2] >= N) {
            b_compat = true; trans_B = CUBLAS_OP_N; ldb = view_B.strides[ndim_b - 2];
        } else if (view_B.strides[ndim_b - 2] == 1 && view_B.strides[ndim_b - 1] >= K) {
            b_compat = true; trans_B = CUBLAS_OP_T; ldb = view_B.strides[ndim_b - 1];
        }

        auto get_offset = [&](int flat_idx, const TensorView& target_view) {
            long long offset = 0;
            int temp = flat_idx;
            for (int i = (int)out_batch_shape.size() - 1; i >= 0; --i) {
                int coord = temp % out_batch_shape[i];
                temp /= out_batch_shape[i];
                int target_dim_idx = i - (out_batch_shape.size() - (target_view.shape.size() - 2));
                if (target_dim_idx >= 0 && target_view.shape[target_dim_idx] > 1) {
                    offset += coord * target_view.strides[target_dim_idx];
                }
            }
            return offset;
        };

        long long stride_A = 0; bool simple_stride_A = true;
        if (batch_count > 1) {
            long long expected_stride = -1;
            for (int b = 0; b < batch_count; ++b) {
                long long offset = get_offset(b, view_A);
                if (b == 0) continue;
                if (b == 1) expected_stride = offset;
                else if (offset != b * expected_stride) { simple_stride_A = false; break; }
            }
            if (simple_stride_A) stride_A = expected_stride;
        }

        long long stride_B = 0; bool simple_stride_B = true;
        if (batch_count > 1) {
            long long expected_stride = -1;
            for (int b = 0; b < batch_count; ++b) {
                long long offset = get_offset(b, view_B);
                if (b == 0) continue;
                if (b == 1) expected_stride = offset;
                else if (offset != b * expected_stride) { simple_stride_B = false; break; }
            }
            if (simple_stride_B) stride_B = expected_stride;
        }

        std::shared_ptr<Buffer> temp_contig_A;
        std::shared_ptr<Buffer> temp_contig_B;

        if (!a_compat || !simple_stride_A) {
            temp_contig_A = empty({batch_count, M, K}, A->device, A->dtype);
            char* a_contig_ptr = static_cast<char*>(temp_contig_A->ptr);

            for (int b = 0; b < batch_count; ++b) {
                long long offset = get_offset(b, view_A);
                dim3 threads = make_copy2d_block();
                dim3 blocks = make_copy2d_grid(K, M, threads);

                if (A->dtype == DType::Float32) {
                    copy_2d_batch<float><<<blocks, threads, 0, cuda::get_stream()>>>(
                        reinterpret_cast<const float*>(a_ptr_raw) + offset,
                        reinterpret_cast<float*>(a_contig_ptr) + (b * M * K),
                        M,
                        K,
                        view_A.strides[ndim_a-2],
                        view_A.strides[ndim_a-1]
                    );
                    CUDART_SAFE_CALL(cudaPeekAtLastError());
                } else if (A->dtype == DType::Float16) {
                    copy_2d_batch<__half><<<blocks, threads, 0, cuda::get_stream()>>>(
                        reinterpret_cast<const __half*>(a_ptr_raw) + offset,
                        reinterpret_cast<__half*>(a_contig_ptr) + (b * M * K),
                        M,
                        K,
                        view_A.strides[ndim_a-2],
                        view_A.strides[ndim_a-1]
                    );
                    CUDART_SAFE_CALL(cudaPeekAtLastError());
                }
            }

            a_ptr_raw = a_contig_ptr;
            trans_A = CUBLAS_OP_N;
            lda = K;
            stride_A = M * K;
            simple_stride_A = true;
            a_compat = true;
        }

        if (!b_compat || !simple_stride_B) {
            temp_contig_B = empty({batch_count, K, N}, A->device, A->dtype);
            char* b_contig_ptr = static_cast<char*>(temp_contig_B->ptr);

            for (int b = 0; b < batch_count; ++b) {
                long long offset = get_offset(b, view_B);
                dim3 threads = make_copy2d_block();
                dim3 blocks = make_copy2d_grid(N, K, threads);

                if (A->dtype == DType::Float32) {
                    copy_2d_batch<float><<<blocks, threads, 0, cuda::get_stream()>>>(
                        reinterpret_cast<const float*>(b_ptr_raw) + offset,
                        reinterpret_cast<float*>(b_contig_ptr) + (b * K * N),
                        K,
                        N,
                        view_B.strides[ndim_b-2],
                        view_B.strides[ndim_b-1]
                    );
                    CUDART_SAFE_CALL(cudaPeekAtLastError());
                } else if (A->dtype == DType::Float16) {
                    copy_2d_batch<__half><<<blocks, threads, 0, cuda::get_stream()>>>(
                        reinterpret_cast<const __half*>(b_ptr_raw) + offset,
                        reinterpret_cast<__half*>(b_contig_ptr) + (b * K * N),
                        K,
                        N,
                        view_B.strides[ndim_b-2],
                        view_B.strides[ndim_b-1]
                    );
                    CUDART_SAFE_CALL(cudaPeekAtLastError());
                }
            }

            b_ptr_raw = b_contig_ptr;
            trans_B = CUBLAS_OP_N;
            ldb = N;
            stride_B = K * N;
            simple_stride_B = true;
            b_compat = true;
        }

        cublasHandle_t handle = cuda::get_cublas_handle();
        cudaDataType_t cuda_type = (A->dtype == DType::Float32) ? CUDA_R_32F : CUDA_R_16F;
        cublasComputeType_t compute_type;

        if (A->dtype == DType::Float32) {
            compute_type = cuda::supports_tf32()
                ? CUBLAS_COMPUTE_32F_FAST_TF32
                : CUBLAS_COMPUTE_32F;
        } else if (A->dtype == DType::Float16) {
            compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
        } else {
            ORMI_THROW(DTypeError, "CUDA MATMUL only supports Float32/Float16.");
        }

        float alpha = 1.0f;
        float beta = 0.0f;

        if (batch_count == 1) {
            CUBLAS_SAFE_CALL(cublasGemmEx(
                handle, (cublasOperation_t)trans_B, (cublasOperation_t)trans_A,
                N, M, K, &alpha,
                b_ptr_raw, cuda_type, ldb,
                a_ptr_raw, cuda_type, lda,
                &beta,
                c_ptr_raw, cuda_type, N,
                compute_type, CUBLAS_GEMM_DEFAULT
            ));
        } else {
            long long stride_C = M * N;
            CUBLAS_SAFE_CALL(cublasGemmStridedBatchedEx(
                handle, (cublasOperation_t)trans_B, (cublasOperation_t)trans_A,
                N, M, K, &alpha,
                b_ptr_raw, cuda_type, ldb, stride_B,
                a_ptr_raw, cuda_type, lda, stride_A,
                &beta,
                c_ptr_raw, cuda_type, N, stride_C,
                batch_count, compute_type, CUBLAS_GEMM_DEFAULT
            ));
        }

        return out_buffer;
    }

    void matmul_handler(const std::shared_ptr<Node>& root) {
        if (!root->data) {
            root->data = empty(root->view.shape, root->device, root->dtype);
        }

        cuda::matmul(
            root->parents[0]->data,
            root->parents[0]->view,
            root->parents[1]->data,
            root->parents[1]->view,
            root->view,
            root->data
        );
    }
    static RegisterGraphBreaker reg_cuda_matmul(DeviceEnum::CUDA, ops::OpEnum::MATMUL, matmul_handler);
}