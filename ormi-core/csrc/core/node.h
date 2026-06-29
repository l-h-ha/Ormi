#pragma once
#include <vector>
#include <memory>

#include "shape.h"
#include "ops.h"
#include "device.h"
#include "dtype.h"
#include "buffer.h"
#include "attributes.h"

namespace ormi {
    struct Node : public std::enable_shared_from_this<Node> {
        ops::OpEnum op;
        std::vector<std::shared_ptr<Node>> parents;

        std::shared_ptr<Buffer> data;
        std::string name;

        bool requires_grad = false;
        Device device;
        DType dtype;
        TensorView view;
        OpAttributes attrs;

        // Leaf node constructor
        Node(std::shared_ptr<Buffer> mem, bool req_grad, std::string name = "");
        Node(std::shared_ptr<Buffer> mem, TensorView view, bool req_grad, std::string name = "");

        // Internal node constructor
        Node(
            ops::OpEnum operation,
            std::vector<std::shared_ptr<Node>> parent_nodes,
            OpAttributes attributes = NoAttrs{},
            std::string name = ""
        );

        bool is_realized() const;
        void realize();
        void set_name(std::string name);
    };

    //
    //
    //

    std::shared_ptr<Node> full(const std::vector<int>& shape, float fill_value, Device device, DType dtype, bool requires_grad);
    std::shared_ptr<Node> scalar(float value, Device device, DType dtype, bool requires_grad);
    std::shared_ptr<Node> zeros(const std::vector<int>& shape, Device device, DType dtype, bool requires_grad);
    std::shared_ptr<Node> ones(const std::vector<int>& shape, Device device, DType dtype, bool requires_grad);

    std::shared_ptr<Node> zeros_like(const std::shared_ptr<Node>& array, DType dtype, bool requires_grad);
    std::shared_ptr<Node> zeros_like(const std::shared_ptr<Node>& array, Device device, DType dtype, bool requires_grad);

    std::shared_ptr<Node> ones_like(const std::shared_ptr<Node>& array, DType dtype, bool requires_grad);
    std::shared_ptr<Node> ones_like(const std::shared_ptr<Node>& array, Device device, DType dtype, bool requires_grad);

    std::shared_ptr<Node> scalar_like(const std::shared_ptr<Node>& array, float value, bool requires_grad = false);
    std::shared_ptr<Node> scalar_like(const std::shared_ptr<Node>& array, float value, Device device, bool requires_grad = false);

    //
    //
    //

    inline std::ostream& operator<<(std::ostream& os, const Node& node) {
        os << "Node(op=" << to_string(node.op)
           << ", dtype=" << to_string(node.dtype)
           << ", requires_grad=" << std::boolalpha << node.requires_grad << std::noboolalpha;

        if (!node.name.empty()) {
            os << ", name=" << node.name;
        }

        if (node.data) {
            bool is_multidimensional = node.data->shape.size() > 1;
            bool is_long_1d = (node.data->shape.size() == 1 && node.data->shape[0] > 10);

            if (is_multidimensional || is_long_1d) {
                os << ",\n     data=\n" << *node.data << ")";
            } else {
                os << ", data=" << *node.data << ")";
            }
        } else {
            if (node.is_realized()) {
                os << ", data=null)";
            } else {
                os << ", data=unrealized)";
            }
        }

        return os;
    }

    inline std::ostream& operator<<(std::ostream& os, const std::shared_ptr<Node>& node) {
        if (!node) {
            return os << "Node(null)";
        }
        return os << *node;
    }

    inline std::string to_string(const Node& node) {
        std::ostringstream oss;
        oss << node;
        return oss.str();
    }

    inline std::string to_string(const std::shared_ptr<Node>& node) {
        std::ostringstream oss;
        oss << node;
        return oss.str();
    }
}