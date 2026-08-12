#include <ormi/core/device.h>
#include <ormi/backend/api.h>

#include "backends/cpu/allocator.h"
#include "backends/cpu/backend.h"
#include "backends/cpu/cast.h"
#include "backends/cpu/fractional.h"
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
            .memory = {
                .allocator = get_allocator,
                .synchronize = synchronize,
                .copy = copy,
                .direct_copy = nullptr,
                .copy_async = copy_async,
                .fill = fill,
                .fill_scalar = fill_scalar
            },
            .ops = {
                .binary = pointwise_binary,
                .unary = pointwise_unary,
                .cast = cast,
                .matmul = matmul,
                .reduce = reduce,
                .all_finite = all_finite,
                .sgd = sgd
            },
            .fractional = {
                .execute = fractional,
                .prepare = nullptr
            },
            .fusion = {
                .supports = jit::supports,
                .prepare = jit::prepare,
                .prepare_many = jit::prepare_many,
                .execute = jit::execute
            },
            .recording = {
                .record = nullptr,
                .replay = nullptr,
                .release = nullptr
            }
        };
    }
}
