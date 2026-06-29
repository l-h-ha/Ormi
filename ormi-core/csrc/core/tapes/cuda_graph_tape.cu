#include "cuda_graph_tape.cuh"
#include "core/error.h"

#include <cuda_runtime.h>
#include "backend/cuda/utils.cuh"
#include "utils/macros.h"

namespace ormi {
    struct CudaGraphTape::Impl {
        cudaGraph_t graph;
        cudaGraphExec_t instance;
        cudaStream_t stream;
        bool initialized = false;
    };

    CudaGraphTape::CudaGraphTape() : pimpl(std::make_unique<Impl>()) {
        pimpl->stream = cuda::get_stream();
    }

    CudaGraphTape::~CudaGraphTape() {
        if (pimpl->initialized) {
            cudaGraphExecDestroy(pimpl->instance);
            cudaGraphDestroy(pimpl->graph);
        }
    }

    void CudaGraphTape::capture(std::function<void()> step_fn) {
        cudaGraph_t new_graph;
        CUDART_SAFE_CALL(cudaStreamBeginCapture(pimpl->stream, cudaStreamCaptureModeGlobal));
        
        step_fn();
        
        CUDART_SAFE_CALL(cudaStreamEndCapture(pimpl->stream, &new_graph));

        if (pimpl->initialized) {
            cudaGraphExecUpdateResult update_result;
            cudaGraphNode_t error_node;
            cudaError_t update_err = cudaGraphExecUpdate(pimpl->instance, new_graph, &error_node, &update_result);
            
            if (update_err == cudaSuccess && update_result == cudaGraphExecUpdateSuccess) {
                CUDART_SAFE_CALL(cudaGraphDestroy(new_graph));
            } else {
                cudaGetLastError(); 
                
                CUDART_SAFE_CALL(cudaGraphExecDestroy(pimpl->instance));
                CUDART_SAFE_CALL(cudaGraphDestroy(pimpl->graph));
                
                pimpl->graph = new_graph;
                CUDART_SAFE_CALL(cudaGraphInstantiate(&pimpl->instance, pimpl->graph, NULL, NULL, 0));
            }
        } else {
            pimpl->graph = new_graph;
            CUDART_SAFE_CALL(cudaGraphInstantiate(&pimpl->instance, pimpl->graph, NULL, NULL, 0));
            pimpl->initialized = true;
        }
    }

    void CudaGraphTape::execute() {
        if (!pimpl->initialized) return;
        CUDART_SAFE_CALL(cudaGraphLaunch(pimpl->instance, pimpl->stream));
    }

    size_t CudaGraphTape::node_count() const {
        if (!pimpl->initialized) return 0;

        size_t count = 0;
        CUDART_SAFE_CALL(cudaGraphGetNodes(pimpl->graph, nullptr, &count));
        return count;
    }
}