#include <cstddef>

#include <cuda_runtime_api.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>

#include "backends/cuda/check.cuh"
#include "backends/cuda/memory.h"
#include "backends/cuda/state.h"

namespace ormi::backends::cuda {
    void copy_async(
        core::Device device,
        void* dst,
        const void* src,
        std::size_t bytes
    ) {
        if (bytes == 0) {
            return;
        }

        auto& cuda_state = state(device);
        check(cudaSetDevice(cuda_state.index()), "cudaSetDevice");
        check(
            cudaMemcpyAsync(
                dst,
                src,
                bytes,
                cudaMemcpyDeviceToDevice,
                cuda_state.stream()
            ),
            "cudaMemcpyAsyncDeviceToDevice"
        );
    }

    void copy(
        core::runtime::Transfer transfer,
        core::Device dst_device,
        void* dst,
        core::Device src_device,
        const void* src,
        std::size_t bytes
    ) {
        if (bytes == 0) {
            return;
        }

        switch (transfer) {
            case core::runtime::Transfer::within: {
                if (src_device.index() != dst_device.index()) {
                    auto& src_state = state(src_device);
                    auto& dst_state = state(dst_device);

                    check(cudaSetDevice(src_state.index()), "cudaSetDevice");
                    check(
                        cudaStreamSynchronize(src_state.stream()),
                        "cudaStreamSynchronize"
                    );

                    check(cudaSetDevice(dst_state.index()), "cudaSetDevice");
                    check(
                        cudaMemcpyPeerAsync(
                            dst,
                            dst_state.index(),
                            src,
                            src_state.index(),
                            bytes,
                            dst_state.stream()
                        ),
                        "cudaMemcpyPeerAsync"
                    );
                    check(
                        cudaStreamSynchronize(dst_state.stream()),
                        "cudaStreamSynchronize"
                    );
                    return;
                }

                auto& cuda_state = state(dst_device);
                copy_async(dst_device, dst, src, bytes);
                check(
                    cudaStreamSynchronize(cuda_state.stream()),
                    "cudaStreamSynchronize"
                );
                return;
            }
            case core::runtime::Transfer::from_host: {
                auto& cuda_state = state(dst_device);
                check(cudaSetDevice(cuda_state.index()), "cudaSetDevice");
                check(
                    cudaMemcpyAsync(
                        dst,
                        src,
                        bytes,
                        cudaMemcpyHostToDevice,
                        cuda_state.stream()
                    ),
                    "cudaMemcpyAsyncHostToDevice"
                );
                check(
                    cudaStreamSynchronize(cuda_state.stream()),
                    "cudaStreamSynchronize"
                );
                return;
            }
            case core::runtime::Transfer::to_host: {
                auto& cuda_state = state(src_device);
                check(cudaSetDevice(cuda_state.index()), "cudaSetDevice");
                check(
                    cudaMemcpyAsync(
                        dst,
                        src,
                        bytes,
                        cudaMemcpyDeviceToHost,
                        cuda_state.stream()
                    ),
                    "cudaMemcpyAsyncDeviceToHost"
                );
                check(
                    cudaStreamSynchronize(cuda_state.stream()),
                    "cudaStreamSynchronize"
                );
                return;
            }
        }

        ORMI_THROW(DeviceError, "CUDA received an invalid memory transfer route.");
    }

    void fill(
        core::Device device,
        void* dst,
        unsigned char value,
        std::size_t bytes
    ) {
        if (bytes == 0) {
            return;
        }

        auto& cuda_state = state(device);
        check(cudaSetDevice(cuda_state.index()), "cudaSetDevice");
        check(
            cudaMemsetAsync(dst, value, bytes, cuda_state.stream()),
            "cudaMemsetAsync"
        );
    }

    void synchronize(core::Device device) {
        auto& cuda_state = state(device);
        check(cudaSetDevice(cuda_state.index()), "cudaSetDevice");
        check(cudaStreamSynchronize(cuda_state.stream()), "cudaStreamSynchronize");
    }
}
