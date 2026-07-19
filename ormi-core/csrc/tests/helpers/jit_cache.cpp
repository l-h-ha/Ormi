#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <string_view>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace {
    void environment(const char* name, const char* value) {
#if defined(_WIN32)
        _putenv_s(name, value);
#else
        if (*value) {
            setenv(name, value, 1);
        } else {
            unsetenv(name);
        }
#endif
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        return 2;
    }

    environment("ORMI_CPU_JIT_CACHE_DIR", argv[1]);
    if (std::string_view(argv[2]) == "load") {
        environment("ORMI_CPU_JIT_CXX", "Z:/ormi/missing/compiler.exe");
        environment("ORMI_CPU_JIT_CXX_ID", "gcc");
    } else {
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
    }

    try {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{1},
            ormi::core::DType::float32
        );
        *static_cast<float*>(input->data()) = 0.5f;
        const auto output = ormi::core::ops::exp(
            ormi::core::ops::mul(input, input)
        );
        const auto value = *static_cast<const float*>(output->data());
        return std::abs(value - std::exp(0.25f)) < 1e-6f ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 4;
    }
}
