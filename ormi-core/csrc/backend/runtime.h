#pragma once
#include <memory>
#include "core/node.h"
#include "core/device.h"

namespace ormi {
    void execute_op(const std::shared_ptr<Node>& root);
    void execute_graph(const std::shared_ptr<Node>& root);
    void synchronize(DeviceEnum dev_type);
}