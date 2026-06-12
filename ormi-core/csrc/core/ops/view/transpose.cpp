#include <memory>
#include <vector>
#include <stdexcept>

#include "core/node.h"
#include "core/ops.h"
#include "core/attributes.h"
#include "core/registry.h"

namespace ormi {
    namespace ops {
        std::shared_ptr<Node> transpose(std::shared_ptr<Node> node, const std::vector<int>& axes) {
            PermuteAttrs attrs{axes};
            return std::make_shared<Node>(ops::OpEnum::TRANSPOSE, std::vector<std::shared_ptr<Node>>{node}, node->device, attrs);
        }
    }

    std::vector<int> infer_transpose_shape(const std::vector<std::shared_ptr<Node>>& parents, const OpAttributes& attrs) {
        if (parents.empty()) throw std::runtime_error("TRANSPOSE requires 1 parent.");
        const auto& in_shape = parents[0]->view.shape;
        const auto& p_attrs = std::get<PermuteAttrs>(attrs);

        if (in_shape.size() != p_attrs.axes.size()) {
            throw std::runtime_error("TRANSPOSE axes count must match tensor dimensions.");
        }

        std::vector<int> out_shape(in_shape.size());
        for (size_t i = 0; i < p_attrs.axes.size(); ++i) {
            out_shape[i] = in_shape[p_attrs.axes[i]];
        }
        return out_shape;
    }

    void adjoint_transpose(std::shared_ptr<Node> node) {
        auto dz = node->grad;
        auto x = node->parents[0];
        if (!x->requires_grad) return;

        const auto& p_attrs = std::get<PermuteAttrs>(node->attrs);

        std::vector<int> inv_axes(p_attrs.axes.size());
        for (size_t i = 0; i < p_attrs.axes.size(); ++i) {
            inv_axes[p_attrs.axes[i]] = i;
        }

        PermuteAttrs inv_attrs{inv_axes};
        auto local_dx = std::make_shared<Node>(
            ops::OpEnum::TRANSPOSE, std::vector<std::shared_ptr<Node>>{dz}, node->device, inv_attrs
        );

        if (!x->grad) x->grad = local_dx;
        else x->grad = std::make_shared<Node>(ops::OpEnum::ADD, std::vector<std::shared_ptr<Node>>{x->grad, local_dx}, node->device);
    }

    std::vector<int> infer_transpose_strides(const std::vector<std::shared_ptr<Node>>& parents, const OpAttributes& attrs, const std::vector<int>& out_shape) {
        const auto& p_attrs = std::get<PermuteAttrs>(attrs);
        auto parent_strides = parents[0]->view.strides;
        
        std::vector<int> out_strides(parent_strides.size());
        for (size_t i = 0; i < p_attrs.axes.size(); ++i) {
            out_strides[i] = parent_strides[p_attrs.axes[i]];
        }
        return out_strides;
    }

    //
    //
    //

    static RegisterShape reg_shape_transpose(ops::OpEnum::TRANSPOSE, infer_transpose_shape);
    static RegisterAdjoint reg_adj_transpose(ops::OpEnum::TRANSPOSE, adjoint_transpose);
    static RegisterStride reg_stride_transpose(ops::OpEnum::TRANSPOSE, infer_transpose_strides);
    static RegisterViewOp reg_view_transpose(ops::OpEnum::TRANSPOSE);
}