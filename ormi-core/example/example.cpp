#include <iostream>

#include <ormi/core/device.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/shape.h>

int main() {
    namespace core = ormi::core;

    const auto x = core::ops::full(
        core::Shape{2, 2},
        2.0f,
        core::Device::cpu()
    );
    const auto squared = core::ops::mul(x, x);
    const auto total = core::ops::sum(squared);

    std::cout << *static_cast<const float*>(total->data()) << '\n';
}
