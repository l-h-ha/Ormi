#include <stdexcept>
#include "runtime.h"
#include "graph_breakers.h"
#include "core/registry.h"
#include "cpu/jit.h"
#include "cuda/jit.cuh"

namespace ormi {
    static bool is_view_op(ops::OpEnum op) {
        return ViewOpRegistry::get().contains(op);
    }

    void execute_graph(std::shared_ptr<Node> root) {
        if (root->is_realized()) return;
        
        for (auto& parent : root->parents) {
            execute_graph(parent);
        }

        if (is_view_op(root->op)) {
            root->data = root->parents[0]->data;
            return;
        }

        if (try_execute_graph_break(root)) {
            return;
        }
        
        switch (root->device.type) {
            case DeviceEnum::CPU:
                cpu::execute_jit(root);
                break;
            case DeviceEnum::CUDA:
                cuda::execute_jit(root);
                break;
            default:
                throw std::runtime_error("FATAL: Unsupported device for JIT compiler.");
        }
    }
}