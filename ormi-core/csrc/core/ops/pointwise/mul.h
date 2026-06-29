#pragma once
#include <memory>
#include "core/node.h"

namespace ormi::ops {
    std::shared_ptr<Node> mul(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b);
}