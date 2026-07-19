#include <ormi/core/device.h>
#include <ormi/core/runtime/backend.h>

#include "backends/cpu/allocator.h"
#include "backends/cpu/backend.h"
#include "backends/cpu/cast.h"
#include "backends/cpu/jit/kernel.h"
#include "backends/cpu/linalg.h"
#include "backends/cpu/memory.h"
#include "backends/cpu/optimizer.h"
#include "backends/cpu/pointwise.h"
#include "backends/cpu/reduction.h"
#include "backends/cpu/reduction/finite.h"

namespace ormi::backends::cpu {
    void synchronize([[maybe_unused]] core::Device device) {}

    core::runtime::Backend init() {
        return core::runtime::Backend{
            .abi = core::runtime::backend_abi,
            .id = core::runtime::backend_id_cpu,
            .name = "cpu",
            .allocator = get_allocator,
            .synchronize = synchronize,
            .copy = copy,
            .fill = fill,
            .fill_scalar = fill_scalar,
            .cast = cast,
            .sgd = sgd,
            .matmul = matmul,
            .reduce = reduce,
            .all_finite = all_finite,
            .pointwise_binary = pointwise_binary,
            .pointwise_unary = pointwise_unary,
            .fusion = jit::execute
        };
    }
}
