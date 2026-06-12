#pragma once
#include <memory>
#include "core/node.h"

namespace ormi {
    void execute_graph(std::shared_ptr<Node> root);
}