#include <ormi/backend/api.h>
#include <ormi/backend/fusion.h>
#include <ormi/backend/id.h>
#include <ormi/backend/plugin.h>
#include <ormi/core/attrs.h>
#include <ormi/core/autocast.h>
#include <ormi/core/autograd.h>
#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/jit.h>
#include <ormi/core/memory/allocator.h>
#include <ormi/core/memory/buffer.h>
#include <ormi/core/memory/ops.h>
#include <ormi/core/op.h>
#include <ormi/core/ops/cast.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/fractional.h>
#include <ormi/core/ops/kind.h>
#include <ormi/core/ops/linalg.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/optim/grad_scaler.h>
#include <ormi/core/optim/sgd.h>
#include <ormi/core/plan.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

template <typename T>
concept ExposesBackend = requires(const T& value) {
    value.backend();
};

static_assert(!ExposesBackend<ormi::core::Device>);

int main() {
    const auto device = ormi::core::Device::cpu();
    return device.is_host() && device.name() == "cpu" ? 0 : 1;
}
