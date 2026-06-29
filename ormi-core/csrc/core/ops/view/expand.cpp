#include <memory>
#include <vector>
#include <stdexcept>

#include "core/adjoint.h"
#include "core/node.h"
#include "core/ops.h"
#include "core/attributes.h"
#include "core/registry.h"
#include "core/node_allocator.h"
#include "core/grad_context.h"

namespace ormi {
    namespace ops {
        std::shared_ptr<Node> expand(const std::shared_ptr<Node>& node, std::vector<int> shape) {
            ExpandAttrs attrs{std::move(shape)};
            return std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                ops::OpEnum::EXPAND,
                std::vector<std::shared_ptr<Node>>{node},
                std::move(attrs)
            );
        }
    }

    std::vector<int> infer_expand_shape(const std::vector<std::shared_ptr<Node>>& parents, const OpAttributes& attrs) {
        if (parents.empty()) {
            ORMI_THROW(OperationError, "EXPAND requires 1 parent.");
        }
        return std::get<ExpandAttrs>(attrs).shape;
    }

    void adjoint_expand(const std::shared_ptr<Node>& node, GradContext& ctx) {
        const auto& dz = ctx.get(node);

        if (!dz) {
            return;
        }

        const auto& x = node->parents[0];

        if (!x->requires_grad) {
            return;
        }

        auto local_dx = unbroadcast(dz, x->view.shape);
        ctx.add(x, local_dx);
    }

    std::vector<int> infer_expand_strides(const std::vector<std::shared_ptr<Node>>& parents, const OpAttributes& attrs, const std::vector<int>& out_shape) {
        const auto& parent_shape = parents[0]->view.shape;
        const auto& parent_strides = parents[0]->view.strides;

        int ndim_in = parent_shape.size();
        int ndim_out = out_shape.size();
        int offset = ndim_out - ndim_in;

        std::vector<int> out_strides(ndim_out, 0);
        for (int i = 0; i < ndim_out; ++i) {
            if (i < offset) {
                out_strides[i] = 0;
            } else {
                int orig_dim = i - offset;
                if (parent_shape[orig_dim] == 1 && out_shape[i] > 1) {
                    out_strides[i] = 0; // Stretched broadcast dimension
                } else {
                    out_strides[i] = parent_strides[orig_dim]; // Mapped dimension
                }
            }
        }
        return out_strides;
    }


    //
    //
    //

    static RegisterShape reg_shape_expand(ops::OpEnum::EXPAND, infer_expand_shape);
    static RegisterAdjoint reg_adj_expand(ops::OpEnum::EXPAND, adjoint_expand);
    static RegisterStride reg_stride_expand(ops::OpEnum::EXPAND, infer_expand_strides);
    static RegisterViewOp reg_view_expand(ops::OpEnum::EXPAND);
}