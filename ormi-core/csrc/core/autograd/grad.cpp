#include <array>
#include <memory>
#include <span>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ormi/core/autograd.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/tensor.h>

namespace ormi::core {
    namespace {
        struct Frame {
            const std::shared_ptr<Tensor>* node;
            std::size_t next_parent;
        };

        bool same_device(const Tensor& lhs, const Tensor& rhs) {
            return
                lhs.device().backend_id() == rhs.device().backend_id() &&
                lhs.device().index() == rhs.device().index();
        }

        std::vector<std::shared_ptr<Tensor>> topology(const std::shared_ptr<Tensor>& root) {
            std::vector<std::shared_ptr<Tensor>> result;
            std::vector<Frame> stack{{&root, 0}};
            std::unordered_set<const Tensor*> visited;
            visited.insert(root.get());

            while (!stack.empty()) {
                auto& frame = stack.back();
                const auto& node = *frame.node;
                const auto& parents = node->parents();

                if (frame.next_parent == parents.size()) {
                    result.push_back(node);
                    stack.pop_back();
                    continue;
                }

                const auto& parent = parents[frame.next_parent++];
                if (!visited.insert(parent.get()).second) {
                    continue;
                }

                stack.push_back(Frame{&parent, 0});
            }

            return result;
        }

        std::vector<std::shared_ptr<Tensor>> grad_impl(
            const std::shared_ptr<Tensor>& output,
            std::span<const std::shared_ptr<Tensor>> inputs,
            std::shared_ptr<Tensor> seed
        ) {
            if (!output) {
                ORMI_THROW(OperationError, "Gradient output cannot be null.");
            }

            if (!is_floating(output->dtype())) {
                ORMI_THROW(DTypeError, "Autograd supports floating-point tensors only.");
            }

            for (const auto& input : inputs) {
                if (!input) {
                    ORMI_THROW(OperationError, "Gradient inputs cannot contain null tensors.");
                }

                if (!is_floating(input->dtype())) {
                    ORMI_THROW(DTypeError, "Autograd supports floating-point tensors only.");
                }
            }

            if (!seed) {
                if (!output->shape().scalar()) {
                    ORMI_THROW(ShapeError, "A non-scalar gradient output requires an explicit seed.");
                }
                seed = ops::ones_like(output);
            }

            if (seed->shape() != output->shape()) {
                ORMI_THROW(ShapeError, "Gradient seed shape must match the output shape.");
            }

            if (seed->dtype() != output->dtype()) {
                ORMI_THROW(DTypeError, "Gradient seed dtype must match the output dtype.");
            }

            if (!same_device(*seed, *output)) {
                ORMI_THROW(DeviceError, "Gradient seed device must match the output device.");
            }

            Grads gradients;
            gradients.add(output, std::move(seed));
            auto nodes = topology(output);

            for (auto node = nodes.rbegin(); node != nodes.rend(); ++node) {
                const auto upstream = gradients.get(*node);

                if (!upstream || !(*node)->op()) {
                    continue;
                }

                if (!(*node)->op()->adjoint) {
                    ORMI_THROW(OperationError, "Operation has no adjoint: " + std::string((*node)->op()->name));
                }

                (*node)->op()->adjoint(*node, upstream, gradients);
            }

            std::vector<std::shared_ptr<Tensor>> result;
            result.reserve(inputs.size());

            for (const auto& input : inputs) {
                auto gradient = gradients.get(input);
                result.push_back(gradient ? std::move(gradient) : ops::zeros_like(input));
            }

            return result;
        }
    }

    std::vector<std::shared_ptr<Tensor>> grad(
        const std::shared_ptr<Tensor>& output,
        const std::vector<std::shared_ptr<Tensor>>& inputs,
        std::shared_ptr<Tensor> seed
    ) {
        return grad_impl(output, inputs, std::move(seed));
    }

    std::shared_ptr<Tensor> grad(
        const std::shared_ptr<Tensor>& output,
        const std::shared_ptr<Tensor>& input,
        std::shared_ptr<Tensor> seed
    ) {
        const std::array inputs{input};
        auto result = grad_impl(output, inputs, std::move(seed));
        return std::move(result[0]);
    }

    std::shared_ptr<Tensor> diff(
        const std::shared_ptr<Tensor>& output,
        const std::shared_ptr<Tensor>& input,
        std::size_t order
    ) {
        if (!output || !input) {
            ORMI_THROW(OperationError, "Differentiation requires an output and input tensor.");
        }

        auto result = output;

        for (std::size_t current = 0; current < order; ++current) {
            result = grad(result, input);
        }

        return result;
    }
}
