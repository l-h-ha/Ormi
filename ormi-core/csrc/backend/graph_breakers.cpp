#include "backend/graph_breakers.h"

namespace ormi {
    static std::map<std::pair<DeviceEnum, ops::OpEnum>, GraphBreakHandler>& get_breaker_registry() {
        static std::map<std::pair<DeviceEnum, ops::OpEnum>, GraphBreakHandler> registry;
        return registry;
    }

    RegisterGraphBreaker::RegisterGraphBreaker(DeviceEnum dev, ops::OpEnum op, GraphBreakHandler handler) {
        get_breaker_registry()[{dev, op}] = handler;
    }

    bool try_execute_graph_break(std::shared_ptr<Node> root) {
        auto& reg = get_breaker_registry();
        auto it = reg.find({root->device.type, root->op});
        
        if (it != reg.end()) {
            it->second(root);
            return true;
        }
        return false;
    }
}