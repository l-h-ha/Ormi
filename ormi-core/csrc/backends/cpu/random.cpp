#include <atomic>
#include <cstddef>
#include <cstdint>
#include <random>

#include <ormi/core/attrs.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/random.h"

namespace ormi::backends::cpu {
    namespace {
        template <typename T>
        void fill_mask(core::Tensor& out, const core::Bernoulli& attrs, std::uint64_t execution) {
            // Include both 64-bit values. seed + execution would make distinct sequences collide.
            std::seed_seq seed{
                static_cast<std::uint32_t>(attrs.seed),
                static_cast<std::uint32_t>(attrs.seed >> 32),
                static_cast<std::uint32_t>(execution),
                static_cast<std::uint32_t>(execution >> 32)
            };
            std::mt19937_64 generator(seed);
            auto* output = static_cast<T*>(out.data());
            const auto scale = static_cast<T>(1.0 / (1.0 - attrs.probability));
            for (std::size_t i = 0; i < out.numel(); ++i) {
                const auto uniform = std::generate_canonical<double, 53>(generator);
                output[i] = uniform < attrs.probability ? T(0) : scale;
            }
        }
    }

    void bernoulli(core::Tensor& out, const core::Bernoulli& attrs) {
        auto& counter = *static_cast<std::uint64_t*>(attrs.counter->data());
        const auto execution =
            std::atomic_ref<std::uint64_t>(counter).fetch_add(1, std::memory_order_relaxed);
        if (out.dtype() == core::DType::float32) {
            fill_mask<float>(out, attrs, execution);
        } else if (out.dtype() == core::DType::float64) {
            fill_mask<double>(out, attrs, execution);
        } else {
            ORMI_THROW(DTypeError, "Bernoulli masks require Float32 or Float64 storage.");
        }
    }
}
