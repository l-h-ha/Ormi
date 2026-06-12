#pragma once
#include <memory>
#include <vector>
#include "core/node.h"

namespace ormi {
    std::shared_ptr<Node> unbroadcast(std::shared_ptr<Node> grad, const std::vector<int>& target_shape);
}