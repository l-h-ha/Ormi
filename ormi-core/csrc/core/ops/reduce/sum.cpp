#include <stdexcept>
#include <algorithm>

#include "core/ops/reduce/sum.h"
#include "core/node_allocator.h"
#include "core/ops.h"
#include "core/node.h"
#include "core/shape.h"
#include "core/grad_context.h"

namespace ormi {
    namespace ops {
        std::shared_ptr<Node> sum(const std::shared_ptr<Node>& node, std::vector<int> axes, const bool keepdims) {
            ReduceAttrs attrs{std::move(axes), keepdims};
            return std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                ops::OpEnum::SUM,
                std::vector<std::shared_ptr<Node>>{node},
                std::move(attrs)
            );
        }

        std::shared_ptr<Node> sum(const std::shared_ptr<Node>& node, const bool keepdims) {
            return sum(node, std::vector<int>{}, keepdims);
        }
    }

    std::vector<int> infer_sum_shape(const std::vector<std::shared_ptr<Node>>& parents, const OpAttributes& attrs) {
        if (parents.empty()) {
            ORMI_THROW(OperationError, "SUM requires 1 parent.");
        }

        const auto& in_shape = parents[0]->view.shape;
        const auto& r_attrs = std::get<ReduceAttrs>(attrs);

        std::vector<int> out_shape;

        // Global reduction
        if (r_attrs.axes.empty()) {
            if (r_attrs.keepdims) {
                out_shape.assign(in_shape.size(), 1);
            }
            return out_shape;
        }

        // Axis-specific reduction
        for (size_t d = 0; d < in_shape.size(); ++d) {
            bool is_reduced = false;
            for (int ax : r_attrs.axes) {
                if (ax == d) { is_reduced = true; break; }
            }

            if (!is_reduced) {
                out_shape.push_back(in_shape[d]);
            } else if (r_attrs.keepdims) {
                out_shape.push_back(1);
            }
        }

        return out_shape;
    }

    // Z = SUM(X)
    // dX = dZ broadcasted to shape of X
    void adjoint_sum(const std::shared_ptr<Node>& node, GradContext& ctx) {
        const auto& dz = ctx.get(node);

        if (!dz) {
            return;
        }

        const auto& x = node->parents[0];

        if (!x->requires_grad) {
            return;
        }

        const auto& r_attrs = std::get<ReduceAttrs>(node->attrs);
        std::shared_ptr<Node> dz_expanded = dz;

        // If keepdims was false, unsqueeze the reduced axes
        if (!r_attrs.keepdims) {
            std::vector<int> target_shape = x->view.shape;
            if (r_attrs.axes.empty()) {
                std::fill(target_shape.begin(), target_shape.end(), 1);
            } else {
                for (int ax : r_attrs.axes) target_shape[ax] = 1;
            }

            ReshapeAttrs res_attrs{target_shape};
            dz_expanded = std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                ops::OpEnum::RESHAPE,
                std::vector<std::shared_ptr<Node>>{dz_expanded},
                std::move(res_attrs)
            );
        }

        // Broadcast the 1s up to the original dimension sizes of X
        ExpandAttrs exp_attrs{x->view.shape};
        auto local_dx = std::allocate_shared<Node>(
            get_node_allocator<Node>(),
            ops::OpEnum::EXPAND,
            std::vector<std::shared_ptr<Node>>{dz_expanded},
            exp_attrs
        );

        ctx.add(x, local_dx);
    }

    static RegisterShape reg_shape_sum(ops::OpEnum::SUM, infer_sum_shape);
    static RegisterAdjoint reg_adj_sum(ops::OpEnum::SUM, adjoint_sum);
}