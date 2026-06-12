#pragma once
#include <map>
#include <functional>
#include <memory>
#include "core/node.h"

namespace ormi {
    using GraphBreakHandler = std::function<void(std::shared_ptr<Node>)>;

    bool try_execute_graph_break(std::shared_ptr<Node> root);
    
    struct RegisterGraphBreaker {
        RegisterGraphBreaker(DeviceEnum dev, ops::OpEnum op, GraphBreakHandler handler);
    };
}