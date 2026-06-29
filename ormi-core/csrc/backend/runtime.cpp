#include <stdexcept>
#include "runtime.h"
#include "graph_breakers.h"
#include "core/registry.h"
#include "cpu/jit.h"
#include "cuda/jit.cuh"
#include "core/error.h"

#ifdef ORMI_USE_CUDA
#include <cuda_runtime.h>
#include "backend/cuda/utils.cuh"
#include "utils/macros.h"
#endif

namespace ormi {
    static bool is_view_op(ops::OpEnum op) {
        return ViewOpRegistry::get().contains(op);
    }

    void execute_op(const std::shared_ptr<Node>& root) {
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
                ORMI_THROW(
                    DeviceError,
                    "Unrecognized DeviceEnum (" + to_string(root->device) + ")"
                );
        }
    }

    void execute_graph(const std::shared_ptr<Node>& root) {
        if (root->is_realized()) return;

        for (auto& parent : root->parents) {
            execute_graph(parent);
        }

        execute_op(root);
    }

    void synchronize(DeviceEnum dev_type) {
        if (dev_type == DeviceEnum::CPU) {
            return; // CPU is already synchronous
        }

        #ifdef ORMI_USE_CUDA
        else if (dev_type == DeviceEnum::CUDA) {
            CUDART_SAFE_CALL(cudaStreamSynchronize(cuda::get_stream()));
            return;
        }
        #endif

        else {
            ORMI_THROW(
                DeviceError,
                "Cannot synchronize on uncompiled or unrecognized device. (" + to_string(dev_type) + ")"
            );
        }
    }
}