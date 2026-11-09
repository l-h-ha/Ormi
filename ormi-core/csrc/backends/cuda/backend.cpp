#include <span>

#include <ormi/backend/api.h>
#include <ormi/backend/plugin.h>
#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/kind.h>
#include <ormi/core/tensor.h>

#include "backends/cuda/allocator.h"
#include "backends/cuda/backend.h"
#include "backends/cuda/cast.h"
#include "backends/cuda/fractional.h"
#include "backends/cuda/fusion/kernel.h"
#include "backends/cuda/indexing.h"
#include "backends/cuda/linalg.h"
#include "backends/cuda/memory.h"
#include "backends/cuda/optimizer/sgd.h"
#include "backends/cuda/pointwise.h"
#include "backends/cuda/record.h"
#include "backends/cuda/random.h"
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
            .memory =
                {.allocator = get_allocator,
                 .synchronize = ormi::backends::cuda::synchronize,
                 .copy = ormi::backends::cuda::copy,
                 .direct_copy = nullptr,
                 .copy_async = ormi::backends::cuda::copy_async,
                 .fill = ormi::backends::cuda::fill,
                 .fill_scalar = ormi::backends::cuda::fill_scalar},
            .ops =
                {.binary = ormi::backends::cuda::pointwise_binary,
                 .unary = ormi::backends::cuda::pointwise_unary,
                 .cast = ormi::backends::cuda::cast,
                 .matmul = ormi::backends::cuda::matmul,
                 .reduce = ormi::backends::cuda::reduce,
                 .all_finite = ormi::backends::cuda::all_finite,
                 .sgd = ormi::backends::cuda::sgd,
                 .pick = ormi::backends::cuda::pick,
                 .unpick = ormi::backends::cuda::unpick,
                 .bernoulli = ormi::backends::cuda::bernoulli,
                 .scatter_scalar = ormi::backends::cuda::scatter_scalar},
            .fractional =
                {.execute = ormi::backends::cuda::fractional,
                 .prepare = ormi::backends::cuda::prepare_fractional},
            .fusion =
                {.supports = ormi::backends::cuda::fusion::supports,
                 .prepare = ormi::backends::cuda::fusion::prepare,
                 .prepare_many = nullptr,
                 .execute = ormi::backends::cuda::fusion::execute},
            .recording =
                {.record = ormi::backends::cuda::record,
                 .replay = ormi::backends::cuda::replay,
                 .release = ormi::backends::cuda::release}
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
