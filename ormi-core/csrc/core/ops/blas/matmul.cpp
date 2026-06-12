#include <stdexcept>
#include <memory>
#include <algorithm>

#include "core/ops/blas/matmul.h"
#include "core/ops.h"
#include "core/node.h"
#include "core/shape.h"
#include "core/adjoint.h" 
#include "core/registry.h"

namespace ormi {
    namespace ops {
        std::shared_ptr<Node> matmul(std::shared_ptr<Node> A, std::shared_ptr<Node> B) {
            if (A->device.type != B->device.type) {
                throw std::runtime_error("Device mismatch in MatMul.");
            }
    
            return std::make_shared<Node>(
                OpEnum::MATMUL, 
                std::vector<std::shared_ptr<Node>>{A, B}, 
                A->device
            );
        }
    }

    std::vector<int> infer_matmul_shape(const std::vector<std::shared_ptr<Node>>& parents, const OpAttributes& attrs) {
        if (parents.size() < 2) throw std::runtime_error("MATMUL requires 2 parents.");
        
        const auto& shape_a = parents[0]->view.shape;
        const auto& shape_b = parents[1]->view.shape;
        
        int ndim_a = shape_a.size();
        int ndim_b = shape_b.size();
    
        if (ndim_a < 2 || ndim_b < 2) {
            throw std::runtime_error("MATMUL requires operands to have at least 2 dimensions.");
        }
    
        int M = shape_a[ndim_a - 2];
        int K_a = shape_a[ndim_a - 1];
        int K_b = shape_b[ndim_b - 2];
        int N = shape_b[ndim_b - 1];
    
        if (K_a != K_b) {
            throw std::runtime_error("MATMUL inner dimensions must match (K_a != K_b).");
        }
    
        std::vector<int> out_shape;
    
        // Handle batch dimension broadcasting if ndim > 2
        if (ndim_a > 2 || ndim_b > 2) {
            std::vector<int> batch_a(shape_a.begin(), shape_a.end() - 2);
            std::vector<int> batch_b(shape_b.begin(), shape_b.end() - 2);
            
            try {
                out_shape = broadcast_shapes(batch_a, batch_b);
            } catch (const std::runtime_error&) {
                throw std::runtime_error("MATMUL batch dimensions could not be broadcast together.");
            }
        }
        
        out_shape.push_back(M);
        out_shape.push_back(N);
        
        return out_shape;
    }

    // Z = X @ Y
    // dZ/dX = dZ @ Y^T, dZ/dY = X^T @ dZ
    void adjoint_matmul(std::shared_ptr<Node> node) {
        auto dz = node->grad;
        auto x = node->parents[0];
        auto y = node->parents[1];

        // Helper to swap the last two dimensions
        auto get_transpose_axes = [](int ndim) {
            std::vector<int> axes(ndim);
            for (int i = 0; i < ndim; ++i) axes[i] = i;
            if (ndim >= 2) std::swap(axes[ndim - 2], axes[ndim - 1]);
            return axes;
        };

        if (x->requires_grad) {
            // dx = dz @ y.T
            PermuteAttrs y_attrs{get_transpose_axes(y->view.shape.size())};
            auto y_t = std::make_shared<Node>(
                ops::OpEnum::TRANSPOSE, 
                std::vector<std::shared_ptr<Node>>{y}, 
                node->device,
                y_attrs
            );
            
            auto dz_matmul_yt = std::make_shared<Node>(
                ops::OpEnum::MATMUL, 
                std::vector<std::shared_ptr<Node>>{dz, y_t}, 
                node->device
            );
            
            auto local_dx = unbroadcast(dz_matmul_yt, x->view.shape);
            
            if (!x->grad) x->grad = local_dx;
            else x->grad = std::make_shared<Node>(
                ops::OpEnum::ADD, 
                std::vector<std::shared_ptr<Node>>{x->grad, local_dx}, 
                node->device
            );
        }

        if (y->requires_grad) {
            // dy = x.T @ dz
            PermuteAttrs x_attrs{get_transpose_axes(x->view.shape.size())};
            auto x_t = std::make_shared<Node>(
                ops::OpEnum::TRANSPOSE, 
                std::vector<std::shared_ptr<Node>>{x}, 
                node->device,
                x_attrs
            );
            
            auto xt_matmul_dz = std::make_shared<Node>(
                ops::OpEnum::MATMUL, 
                std::vector<std::shared_ptr<Node>>{x_t, dz}, 
                node->device
            );
            
            auto local_dy = unbroadcast(xt_matmul_dz, y->view.shape);
            
            if (!y->grad) y->grad = local_dy;
            else y->grad = std::make_shared<Node>(
                ops::OpEnum::ADD, 
                std::vector<std::shared_ptr<Node>>{y->grad, local_dy}, 
                node->device
            );
        }
    }

    static RegisterShape reg_shape_matmul(ops::OpEnum::MATMUL, infer_matmul_shape);
    static RegisterAdjoint reg_adj_matmul(ops::OpEnum::MATMUL, adjoint_matmul);
}