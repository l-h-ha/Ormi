#include <stdexcept>
#include <memory>
#include <algorithm>

#include "core/ops/blas/matmul.h"
#include "core/ops/reduce/sum.h"
#include "core/ops/pointwise/mul.h"
#include "core/ops.h"
#include "core/node.h"
#include "core/shape.h"
#include "core/adjoint.h"
#include "core/registry.h"
#include "core/grad_context.h"
#include "core/node_allocator.h"


namespace ormi {
    namespace ops {
        std::shared_ptr<Node> matmul(const std::shared_ptr<Node>& A, const std::shared_ptr<Node>& B) {
            if (A->device.type != B->device.type) {
                ORMI_CHECK_EQ(B->device.type, A->device.type, ormi::DeviceError, "Device mismatch in MATMUL.");
            }

            return std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                OpEnum::MATMUL,
                std::vector<std::shared_ptr<Node>>{A, B}
            );
        }
    }

    std::vector<int> infer_matmul_shape(const std::vector<std::shared_ptr<Node>>& parents, const OpAttributes& attrs) {
        if (parents.size() < 2) {
            ORMI_THROW(
                OperationError,
                "MATMUL requires 2 parents. (currently" + std::to_string(parents.size()) + ")"
            );
        }

        const auto& shape_a = parents[0]->view.shape;
        const auto& shape_b = parents[1]->view.shape;

        int ndim_a = shape_a.size();
        int ndim_b = shape_b.size();

        if (ndim_a < 2 || ndim_b < 2) {
            using ormi::detail::operator<<;
            std::ostringstream oss;
            oss << "MATMUL requires both operands to have at least 2 dimensions.\n"
                << "  Shape A: " << shape_a << " (" << ndim_a << "D)\n"
                << "  Shape B: " << shape_b << " (" << ndim_b << "D)";
            ORMI_THROW(ormi::ShapeError, oss.str());
        }

        int M = shape_a[ndim_a - 2];
        int K_a = shape_a[ndim_a - 1];
        int K_b = shape_b[ndim_b - 2];
        int N = shape_b[ndim_b - 1];

        ORMI_CHECK_EQ(K_b, K_a, ormi::ShapeError, "MATMUL inner dimensions must match.");

        std::vector<int> out_shape;

        if (ndim_a > 2 || ndim_b > 2) {
            std::vector<int> batch_a(shape_a.begin(), shape_a.end() - 2);
            std::vector<int> batch_b(shape_b.begin(), shape_b.end() - 2);

            try {
                out_shape = broadcast_shapes(batch_a, batch_b);
            } catch (const ormi::ShapeError&) {
                using ormi::detail::operator<<;
                std::ostringstream oss;

                oss << "MATMUL batch dimensions could not be broadcast together.\n"
                    << "  Full Shape A: " << shape_a << "\n"
                    << "  Full Shape B: " << shape_b;

                ORMI_THROW(ormi::ShapeError, oss.str());
            }
        }

        out_shape.push_back(M);
        out_shape.push_back(N);

        return out_shape;
    }

    // Z = X @ Y
    void adjoint_matmul(const std::shared_ptr<Node>& node, GradContext& ctx) {
        const auto& dz = ctx.get(node);

        if (!dz) {
            return;
        }

        const auto& x = node->parents[0];
        const auto& y = node->parents[1];

        auto get_transpose_axes = [](int ndim) {
            std::vector<int> axes(ndim);
            for (int i = 0; i < ndim; ++i) axes[i] = i;
            if (ndim >= 2) std::swap(axes[ndim - 2], axes[ndim - 1]);
            return axes;
        };

        if (x->requires_grad) {
            std::shared_ptr<Node> local_dx;
            PermuteAttrs y_attrs{get_transpose_axes(y->view.shape.size())};
            auto y_t = std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                ops::OpEnum::TRANSPOSE,
                std::vector<std::shared_ptr<Node>>{y},
                std::move(y_attrs)
            );

            auto dz_matmul_yt = std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                ops::OpEnum::MATMUL,
                std::vector<std::shared_ptr<Node>>{dz, y_t}
            );

            local_dx = unbroadcast(dz_matmul_yt, x->view.shape);
            ctx.add(x, local_dx);
        }

        if (y->requires_grad) {
            std::shared_ptr<Node> local_dy;
            PermuteAttrs x_attrs{get_transpose_axes(x->view.shape.size())};
            auto x_t = std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                ops::OpEnum::TRANSPOSE,
                std::vector<std::shared_ptr<Node>>{x},
                std::move(x_attrs)
            );

            auto xt_matmul_dz = std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                ops::OpEnum::MATMUL,
                std::vector<std::shared_ptr<Node>>{x_t, dz}
            );

            local_dy = unbroadcast(xt_matmul_dz, y->view.shape);
            ctx.add(y, local_dy);
        }
    }

    static RegisterShape reg_shape_matmul(ops::OpEnum::MATMUL, infer_matmul_shape);
    static RegisterAdjoint reg_adj_matmul(ops::OpEnum::MATMUL, adjoint_matmul);
}