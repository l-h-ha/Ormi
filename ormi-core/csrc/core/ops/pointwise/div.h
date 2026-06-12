#pragma once
#include <memory>
#include "core/node.h"

namespace ormi::ops {
    std::shared_ptr<Node> div(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
}