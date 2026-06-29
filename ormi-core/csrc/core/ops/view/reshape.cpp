#include <memory>
#include <vector>
#include <stdexcept>
#include <utility>
#include <sstream>

#include "core/error.h"
#include "core/node.h"
#include "core/ops.h"
#include "core/attributes.h"
#include "core/registry.h"
#include "core/node_allocator.h"
#include "core/grad_context.h"

namespace ormi {
    namespace ops {
        std::shared_ptr<Node> reshape(const std::shared_ptr<Node>& node, std::vector<int> shape) {
            ReshapeAttrs attrs{std::move(shape)};
            return std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                ops::OpEnum::RESHAPE,
                std::vector<std::shared_ptr<Node>>{node},
                std::move(attrs)
            );
        }
    }

    std::vector<int> infer_reshape_shape(const std::vector<std::shared_ptr<Node>>& parents, const OpAttributes& attrs) {
        ORMI_CHECK_EX(!parents.empty(), ormi::ShapeError, "RESHAPE requires 1 parent.");
        return std::get<ReshapeAttrs>(attrs).shape;
    }

    void adjoint_reshape(const std::shared_ptr<Node>& node, GradContext& ctx) {
        const auto& dz = ctx.get(node);

        if (!dz) {
            return;
        }

        const auto& x = node->parents[0];

        if (!x->requires_grad) {
            return;
        }

        ReshapeAttrs res_attrs{x->view.shape};
        auto local_dx = std::allocate_shared<Node>(
            get_node_allocator<Node>(),
            ops::OpEnum::RESHAPE,
            std::vector<std::shared_ptr<Node>>{dz},
            std::move(res_attrs)
        );

        ctx.add(x, local_dx);
    }

    std::vector<int> infer_reshape_strides(const std::vector<std::shared_ptr<Node>>& parents, const OpAttributes& attrs, const std::vector<int>& out_shape) {
        const auto& old_shape = parents[0]->view.shape;
        const auto& old_strides = parents[0]->view.strides;

        if (old_shape.empty()) return calc_contiguous_strides(out_shape);

        size_t old_vol = 1, new_vol = 1;
        for (int d : old_shape) old_vol *= d;
        for (int d : out_shape) new_vol *= d;

        if (old_vol != new_vol) {
            using ormi::detail::operator<<;
            std::ostringstream oss;
            oss << "RESHAPE cannot change the total number of elements.\n"
                << "  Original Shape: " << old_shape << " (Volume: " << old_vol << ")\n"
                << "  Target Shape:   " << out_shape << " (Volume: " << new_vol << ")";
            ORMI_THROW(ormi::ShapeError, oss.str());
        }

        std::vector<int> new_strides(out_shape.size(), 0);
        int o = 0, n = 0;

        while (n < out_shape.size() && o < old_shape.size()) {
            int old_size = old_shape[o];
            int new_size = out_shape[n];
            int o_end = o + 1;
            int n_end = n + 1;

            while (old_size != new_size) {
                if (old_size < new_size) {
                    ORMI_CHECK_EX(o_end < old_shape.size(), ormi::OrmiError, "Internal stride math error.");
                    old_size *= old_shape[o_end++];
                } else {
                    ORMI_CHECK_EX(n_end < out_shape.size(), ormi::OrmiError, "Internal stride math error.");
                    new_size *= out_shape[n_end++];
                }
            }

            for (int i = o; i < o_end - 1; ++i) {
                if (old_shape[i] != 1 && old_shape[i+1] != 1) {
                    if (old_strides[i] != old_shape[i+1] * old_strides[i+1]) {
                        using ormi::detail::operator<<;
                        std::ostringstream oss;
                        oss << "RESHAPE cannot be performed as a zero-copy view. The tensor is not contiguous across the dimensions being merged.\n"
                            << "  Original Shape:   " << old_shape << "\n"
                            << "  Original Strides: " << old_strides << "\n"
                            << "  Target Shape:     " << out_shape;
                        ORMI_THROW(ormi::ShapeError, oss.str());
                    }
                }
            }

            int current_stride = old_strides[o_end - 1];
            for (int i = n_end - 1; i >= n; --i) {
                new_strides[i] = current_stride;
                current_stride *= out_shape[i];
            }

            o = o_end;
            n = n_end;
        }

        return new_strides;
    }

    static RegisterShape reg_shape_reshape(ops::OpEnum::RESHAPE, infer_reshape_shape);
    static RegisterAdjoint reg_adj_reshape(ops::OpEnum::RESHAPE, adjoint_reshape);
    static RegisterStride reg_stride_reshape(ops::OpEnum::RESHAPE, infer_reshape_strides);
    static RegisterViewOp reg_view_reshape(ops::OpEnum::RESHAPE);
}