#include <memory>
#include <vector>
#include <stdexcept>

#include "core/node.h"
#include "core/ops.h"
#include "core/attributes.h"
#include "core/registry.h"

namespace ormi {
    namespace ops {
        std::shared_ptr<Node> reshape(std::shared_ptr<Node> node, const std::vector<int>& shape) {
            ReshapeAttrs attrs{shape};
            return std::make_shared<Node>(ops::OpEnum::RESHAPE, std::vector<std::shared_ptr<Node>>{node}, node->device, attrs);
        }
    }

    std::vector<int> infer_reshape_shape(const std::vector<std::shared_ptr<Node>>& parents, const OpAttributes& attrs) {
        if (parents.empty()) throw std::runtime_error("RESHAPE requires 1 parent.");
        return std::get<ReshapeAttrs>(attrs).shape;
    }

    void adjoint_reshape(std::shared_ptr<Node> node) {
        auto dz = node->grad;
        auto x = node->parents[0];
        if (!x->requires_grad) return;

        ReshapeAttrs res_attrs{x->view.shape};
        auto local_dx = std::make_shared<Node>(
            ops::OpEnum::RESHAPE, std::vector<std::shared_ptr<Node>>{dz}, node->device, res_attrs
        );

        if (!x->grad) x->grad = local_dx;
        else x->grad = std::make_shared<Node>(ops::OpEnum::ADD, std::vector<std::shared_ptr<Node>>{x->grad, local_dx}, node->device);
    }

    std::vector<int> infer_reshape_strides(const std::vector<std::shared_ptr<Node>>& parents, const OpAttributes& attrs, const std::vector<int>& out_shape) {
        const auto& old_shape = parents[0]->view.shape;
        const auto& old_strides = parents[0]->view.strides;
        
        if (old_shape.empty()) return calc_contiguous_strides(out_shape);

        std::vector<int> new_strides(out_shape.size(), 0);
        int o = 0, n = 0;

        while (n < out_shape.size() && o < old_shape.size()) {
            int old_size = old_shape[o];
            int new_size = out_shape[n];
            int o_end = o + 1;
            int n_end = n + 1;

            // Find matching blocks of elements
            while (old_size != new_size) {
                if (old_size < new_size) {
                    if (o_end >= old_shape.size()) throw std::runtime_error("RESHAPE size mismatch.");
                    old_size *= old_shape[o_end++];
                } else {
                    if (n_end >= out_shape.size()) throw std::runtime_error("RESHAPE size mismatch.");
                    new_size *= out_shape[n_end++];
                }
            }

            // Verify contiguity of the memory block being merged
            for (int i = o; i < o_end - 1; ++i) {
                if (old_shape[i] != 1 && old_shape[i+1] != 1) {
                    if (old_strides[i] != old_shape[i+1] * old_strides[i+1]) {
                        throw std::runtime_error("RESHAPE cannot be performed as a zero-copy view because the tensor is not contiguous across merged dimensions.");
                    }
                }
            }

            // Assign strides recursively to the new block
            int current_stride = old_strides[o_end - 1];
            for (int i = n_end - 1; i >= n; --i) {
                new_strides[i] = current_stride;
                current_stride *= out_shape[i];
            }

            o = o_end;
            n = n_end;
        }

        while (o < old_shape.size()) {
            if (old_shape[o++] != 1) throw std::runtime_error("RESHAPE size mismatch trailing dims.");
        }
        while (n < out_shape.size()) {
            if (out_shape[n] != 1) throw std::runtime_error("RESHAPE size mismatch trailing dims.");
            new_strides[n++] = 1;
        }

        return new_strides;
    }

    static RegisterShape reg_shape_reshape(ops::OpEnum::RESHAPE, infer_reshape_shape);
    static RegisterAdjoint reg_adj_reshape(ops::OpEnum::RESHAPE, adjoint_reshape);
    static RegisterStride reg_stride_reshape(ops::OpEnum::RESHAPE, infer_reshape_strides);
    static RegisterViewOp reg_view_reshape(ops::OpEnum::RESHAPE);
}