#pragma once
#include <memory>
#include "core/node.h"

namespace ormi::ops {
    std::shared_ptr<Node> cos(const std::shared_ptr<Node>& a);
}