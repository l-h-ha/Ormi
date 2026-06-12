#pragma once
#include <memory>
#include "core/node.h"

namespace ormi::cpu {
    void execute_jit(std::shared_ptr<Node> root);
}