#include <utility>
#include "core/node.h"
#include "backend/runtime.h"
#include "core/shape.h"
#include "core/registry.h"
#include "core/buffer.h"

namespace ormi {
    // Leaf node constructor
    Node::Node(std::shared_ptr<Buffer> mem, bool req_grad, std::string name)
        : op(ops::OpEnum::LEAF), requires_grad(req_grad), name(name) {

        this->data = std::move(mem);
        this->device = this->data->device;
        this->dtype = this->data->dtype;
        this->view.shape = this->data->shape;
        this->view.strides = calc_contiguous_strides(this->data->shape);
    }

    Node::Node(std::shared_ptr<Buffer> mem, TensorView view, bool req_grad, std::string name)
        : op(ops::OpEnum::LEAF), view(std::move(view)), requires_grad(req_grad), name(name) {

        this->data = std::move(mem);
        this->device = this->data->device;
        this->dtype = this->data->dtype;
    }

    // Internal node constructor
    Node::Node(ops::OpEnum operation, std::vector<std::shared_ptr<Node>> parent_nodes, OpAttributes attributes, std::string name)
        : op(operation), parents(std::move(parent_nodes)), requires_grad(false), attrs(std::move(attributes)), name(name) {

        if (this->parents.empty()) {
            ORMI_THROW(NodeError, "Internal node must have at least 1 parent.");
        }

        Device device = this->parents[0]->device;
        DType dtype = this->parents[0]->dtype;
        for (size_t i = 1; i < this->parents.size(); ++i) {
            const auto& p = this->parents[i];
            ORMI_CHECK_EQ(
                device, p->device,
                ormi::DeviceError, "Parents' devices must match."
            );
            ORMI_CHECK_EQ(
                dtype, p->dtype,
                ormi::DTypeError, "Parents' DTypes must match."
            );
        }
        this->device = device;
        this->dtype = dtype;

        // Inherit requires_grad if any parent requires it
        for (const auto& p : parents) {
            if (p->requires_grad) requires_grad = true;
        }

        this->view.shape = infer_shape(this->op, parents, attrs);

        auto stride_func = Registry<StrideInferFunc>::get(operation);
        if (stride_func) {
            this->view.strides = stride_func(parents, attrs, this->view.shape);
        } else {
            this->view.strides = calc_contiguous_strides(this->view.shape);
        }
    }

    bool Node::is_realized() const {
        return this->data != nullptr;
    }

    void Node::realize() {
        if (!is_realized()) {
            execute_graph(shared_from_this());
        }
    }

    void Node::set_name(std::string name) {
        this->name = name;
    }

    //
    //
    //

    std::shared_ptr<Node> full(const std::vector<int>& shape, float fill_value, Device device, DType dtype, bool requires_grad) {
        return std::make_shared<Node>(full(shape, fill_value, device, dtype), requires_grad);
    }

    std::shared_ptr<Node> scalar(float value, Device device, DType dtype, bool requires_grad) {
        return std::make_shared<Node>(scalar(value, device, dtype), requires_grad);
    }

    std::shared_ptr<Node> zeros(const std::vector<int>& shape, Device device, DType dtype, bool requires_grad) {
        return std::make_shared<Node>(zeros(shape, device, dtype), requires_grad);
    }

    std::shared_ptr<Node> ones(const std::vector<int>& shape, Device device, DType dtype, bool requires_grad) {
        return std::make_shared<Node>(ones(shape, device, dtype), requires_grad);
    }

    std::shared_ptr<Node> zeros_like(const std::shared_ptr<Node>& array, DType dtype, bool requires_grad) {
        return std::make_shared<Node>(
            zeros(array->view.shape, array->device, dtype),
            requires_grad
        );
    }

    std::shared_ptr<Node> zeros_like(const std::shared_ptr<Node>& array, Device device, DType dtype, bool requires_grad) {
        return std::make_shared<Node>(
            zeros(array->view.shape, device, dtype),
            requires_grad
        );
    }

    std::shared_ptr<Node> ones_like(const std::shared_ptr<Node>& array, DType dtype, bool requires_grad) {
        return std::make_shared<Node>(
            ones(array->view.shape, array->device, dtype),
            requires_grad
        );
    }

    std::shared_ptr<Node> ones_like( const std::shared_ptr<Node>& array, Device device, DType dtype, bool requires_grad) {
        return std::make_shared<Node>(
            ones(array->view.shape, device, dtype),
            requires_grad
        );
    }

    std::shared_ptr<Node> scalar_like(const std::shared_ptr<Node>& array, float value, bool requires_grad ) {
        return std::make_shared<Node>(
            scalar(value, array->device, array->dtype),
            requires_grad
        );
    }

    std::shared_ptr<Node> scalar_like(const std::shared_ptr<Node>& array, float value, Device device, bool requires_grad) {
        return std::make_shared<Node>(
            scalar(value, device, array->dtype),
            requires_grad
        );
    }
}