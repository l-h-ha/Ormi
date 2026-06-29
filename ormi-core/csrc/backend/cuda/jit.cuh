#pragma once
#include <memory>
#include "core/node.h"

namespace ormi::cuda {
    void execute_jit(const std::shared_ptr<Node>& root);
}