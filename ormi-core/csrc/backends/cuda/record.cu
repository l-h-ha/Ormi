#include <memory>

#include <cuda_runtime_api.h>

#include <ormi/core/device.h>
#include <ormi/core/error.h>
#include <ormi/backend/api.h>

#include "backends/cuda/check.cuh"
#include "backends/cuda/record.h"
#include "backends/cuda/state.h"

namespace ormi::backends::cuda {
    namespace {
        class Recording final {
        public:
            Recording(
                int device,
                cudaStream_t stream
            ) : device_(device), stream_(stream) {}

            ~Recording() noexcept {
                if (cudaSetDevice(device_) != cudaSuccess) {
                    return;
                }
                if (executable_) {
                    (void)cudaGraphExecDestroy(executable_);
                }
                if (graph_) {
                    (void)cudaGraphDestroy(graph_);
                }
            }

            Recording(const Recording&) = delete;
            Recording& operator=(const Recording&) = delete;

            void finish(cudaGraph_t graph) {
                graph_ = graph;
                check(
                    cudaGraphInstantiateWithFlags(
                        &executable_,
                        graph_,
                        0
                    ),
                    "cudaGraphInstantiateWithFlags"
                );
            }

            void run() {
                check(cudaSetDevice(device_), "cudaSetDevice");
                check(
                    cudaGraphLaunch(executable_, stream_),
                    "cudaGraphLaunch"
                );
            }

        private:
            int device_;
            cudaStream_t stream_;
            cudaGraph_t graph_ = nullptr;
            cudaGraphExec_t executable_ = nullptr;
        };

        void discard(cudaStream_t stream) noexcept {
            cudaGraph_t graph = nullptr;
            (void)cudaStreamEndCapture(stream, &graph);
            if (graph) {
                (void)cudaGraphDestroy(graph);
            }
        }
    }

    void* record(
        core::Device device,
        core::runtime::RecordBodyFn body,
        void* context
    ) {
        if (!body) {
            ORMI_THROW(BackendError, "Recording requires an execution body.");
        }

        auto& current = state(device);
        const auto stream = current.stream();
        auto result = std::make_unique<Recording>(
            device.index(),
            stream
        );
        check(
            cudaStreamBeginCapture(
                stream,
                cudaStreamCaptureModeThreadLocal
            ),
            "cudaStreamBeginCapture"
        );

        try {
            body(context);
        } catch (...) {
            discard(stream);
            throw;
        }

        cudaGraph_t graph = nullptr;
        const auto status = cudaStreamEndCapture(stream, &graph);
        if (status != cudaSuccess) {
            if (graph) {
                (void)cudaGraphDestroy(graph);
            }
            check(status, "cudaStreamEndCapture");
        }
        if (!graph) {
            ORMI_THROW(BackendError, "CUDA recording produced no graph.");
        }

        result->finish(graph);
        return result.release();
    }

    void replay(void* recording) {
        if (!recording) {
            ORMI_THROW(BackendError, "Cannot replay an empty recording.");
        }
        static_cast<Recording*>(recording)->run();
    }

    void release(void* recording) noexcept {
        delete static_cast<Recording*>(recording);
    }
}
