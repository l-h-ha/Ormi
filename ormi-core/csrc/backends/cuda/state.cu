#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <ormi/core/device.h>
#include <ormi/core/error.h>

#include "backends/cuda/check.cuh"
#include "backends/cuda/state.h"
#include "backends/cuda/utils.cuh"

namespace ormi::backends::cuda {
    State::State(int index) : index_(index) {
        check_device_id(index);
        check(cudaSetDevice(index_), "cudaSetDevice");
        check(
            cudaDeviceGetAttribute(
                &compute_major_,
                cudaDevAttrComputeCapabilityMajor,
                index_
            ),
            "cudaDeviceGetAttributeComputeCapabilityMajor"
        );
        check(
            cudaDeviceGetAttribute(
                &compute_minor_,
                cudaDevAttrComputeCapabilityMinor,
                index_
            ),
            "cudaDeviceGetAttributeComputeCapabilityMinor"
        );
        check(
            cudaDeviceGetAttribute(
                &multiprocessors_,
                cudaDevAttrMultiProcessorCount,
                index_
            ),
            "cudaDeviceGetAttributeMultiProcessorCount"
        );

        check(
            cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags"
        );
        const auto cublas_status = cublasCreate(&cublas_);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            (void)cudaStreamDestroy(stream_);
            stream_ = nullptr;
            check(cublas_status, "cublasCreate");
        }

        const auto stream_status = cublasSetStream(cublas_, stream_);
        if (stream_status != CUBLAS_STATUS_SUCCESS) {
            (void)cublasDestroy(cublas_);
            (void)cudaStreamDestroy(stream_);
            cublas_ = nullptr;
            stream_ = nullptr;
            check(stream_status, "cublasSetStream");
        }

    #if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
        int supported = 0;
        const cudaError_t capability_status = cudaDeviceGetAttribute(
            &supported,
            cudaDevAttrMemoryPoolsSupported,
            index_
        );

        if (capability_status == cudaSuccess && supported != 0) {
            cudaMemPoolProps properties{};
            properties.allocType = cudaMemAllocationTypePinned;
            properties.handleTypes = cudaMemHandleTypeNone;
            properties.location.type = cudaMemLocationTypeDevice;
            properties.location.id = index_;

            cudaError_t status = cudaMemPoolCreate(&pool_, &properties);

            if (status != cudaSuccess) {
                (void)cublasDestroy(cublas_);
                (void)cudaStreamDestroy(stream_);
                cublas_ = nullptr;
                stream_ = nullptr;
                check(status, "cudaMemPoolCreate");
            }

            std::uint64_t release_threshold =
                std::numeric_limits<std::uint64_t>::max();
            status = cudaMemPoolSetAttribute(
                pool_,
                cudaMemPoolAttrReleaseThreshold,
                &release_threshold
            );

            if (status != cudaSuccess) {
                (void)cudaMemPoolDestroy(pool_);
                (void)cublasDestroy(cublas_);
                (void)cudaStreamDestroy(stream_);
                pool_ = nullptr;
                cublas_ = nullptr;
                stream_ = nullptr;
                check(status, "cudaMemPoolSetAttribute");
            }

            has_pool_ = true;
        }
    #endif
    }

    State::~State() noexcept {
        if (!stream_) {
            return;
        }

        if (cudaSetDevice(index_) != cudaSuccess) {
            return;
        }

        (void)cudaStreamSynchronize(stream_);

    #if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
        if (pool_) {
            (void)cudaMemPoolDestroy(pool_);
        }
    #endif

        (void)cublasDestroy(cublas_);
        (void)cudaStreamDestroy(stream_);
    }

    int State::index() const noexcept {
        return index_;
    }

    int State::compute_major() const noexcept {
        return compute_major_;
    }

    int State::compute_minor() const noexcept {
        return compute_minor_;
    }

    int State::multiprocessors() const noexcept {
        return multiprocessors_;
    }

    cudaStream_t State::stream() const noexcept {
        return stream_;
    }

    cublasHandle_t State::cublas() const noexcept {
        return cublas_;
    }

    bool State::has_pool() const noexcept {
        return has_pool_;
    }

#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
    cudaMemPool_t State::pool() const noexcept {
        return pool_;
    }
#endif

    State& state(core::Device device) {
        if (device.name() != "cuda") {
            ORMI_THROW(BackendError, "CUDA state requires a CUDA device.");
        }

        static const int device_cnt = available_devices();
        static std::mutex mutex;
        static std::vector<std::unique_ptr<State>> states(
            static_cast<std::size_t>(device_cnt)
        );

        const int index = device.index();
        check_device_id(index);

        std::lock_guard lock(mutex);

        auto& slot = states[static_cast<std::size_t>(index)];

        if (!slot) {
            slot = std::make_unique<State>(index);
        }

        return *slot;
    }

    cudaStream_t stream(core::Device device) {
        return state(device).stream();
    }
}
