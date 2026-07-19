#include <span>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/kind.h>
#include <ormi/core/runtime/backend.h>
#include <ormi/core/runtime/plugin.h>
#include <ormi/core/tensor.h>

#include "backends/cuda/backend.h"
#include "backends/cuda/allocator.h"
#include "backends/cuda/cast.h"
#include "backends/cuda/fusion/kernel.h"
#include "backends/cuda/linalg.h"
#include "backends/cuda/memory.h"
#include "backends/cuda/optimizer/sgd.h"
#include "backends/cuda/pointwise.h"
#include "backends/cuda/reduction.h"
#include "backends/cuda/reduction/finite.h"

namespace ormi::backends::cuda {
    namespace {
        core::memory::Allocator* get_allocator(core::Device device) {
            return allocator(device);
        }

    }

    core::runtime::Backend init() {
        return core::runtime::Backend{
            .abi = core::runtime::backend_abi,
            .id = core::runtime::backend_id_cuda,
            .name = "cuda",
            .allocator = get_allocator,
            .synchronize = ormi::backends::cuda::synchronize,
            .copy = ormi::backends::cuda::copy,
            .fill = ormi::backends::cuda::fill,
            .fill_scalar = ormi::backends::cuda::fill_scalar,
            .cast = ormi::backends::cuda::cast,
            .sgd = ormi::backends::cuda::sgd,
            .matmul = ormi::backends::cuda::matmul,
            .reduce = ormi::backends::cuda::reduce,
            .all_finite = ormi::backends::cuda::all_finite,
            .pointwise_binary = ormi::backends::cuda::pointwise_binary,
            .pointwise_unary = ormi::backends::cuda::pointwise_unary,
            .fusion = ormi::backends::cuda::fusion::execute
        };
    }
}

#if defined(_WIN32)
#define ORMI_PLUGIN_EXPORT extern "C" __declspec(dllexport)
#else
#define ORMI_PLUGIN_EXPORT extern "C" __attribute__((visibility("default")))
#endif

ORMI_PLUGIN_EXPORT const ormi::core::runtime::Backend* ormi_backend_init() noexcept {
    static const auto backend = ormi::backends::cuda::init();
    return &backend;
}

#undef ORMI_PLUGIN_EXPORT
