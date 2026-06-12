#pragma once
#include <memory>
#include "core/node.h"

namespace ormi::ops {
    std::shared_ptr<Node> sub(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
}