#pragma once
#include <memory>
#include "node.h"

namespace ormi {
    void compute_gradients(std::shared_ptr<Node> root);
    void zero_grad_graph(std::shared_ptr<Node> root);
}