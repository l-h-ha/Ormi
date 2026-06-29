#pragma once
#include <memory>
#include "grad_context.h"
#include "node.h"

namespace ormi {
    std::vector<std::shared_ptr<Node>> grad(
        const std::shared_ptr<Node>& root,
        const std::vector<std::shared_ptr<Node>>& inputs
    );

    std::vector<std::shared_ptr<Node>> grad(
        const std::shared_ptr<Node>& root,
        const std::vector<std::shared_ptr<Node>>& inputs,
        const std::shared_ptr<Node>& seed
    );

    std::shared_ptr<Node> diff(
        const std::shared_ptr<Node>& y,
        const std::shared_ptr<Node>& x,
        int order = 1
    );
}