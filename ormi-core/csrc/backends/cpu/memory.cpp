#include <cstdint>
#include <cstring>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>

#include "backends/cpu/memory.h"

namespace ormi::backends::cpu {
    void copy(
        [[maybe_unused]] core::runtime::Transfer transfer,
        [[maybe_unused]] core::Device dst_device,
        void* dst,
        [[maybe_unused]] core::Device src_device,
        const void* src,
        std::size_t bytes
    ) {
        std::memcpy(dst, src, bytes);
    }

    void copy_async(
        [[maybe_unused]] core::Device device,
        void* dst,
        const void* src,
        std::size_t bytes
    ) {
        std::memcpy(dst, src, bytes);
    }

    void fill([[maybe_unused]] core::Device device, void* dst, unsigned char value, std::size_t bytes) {
        std::memset(dst, value, bytes);
    }

    namespace {
        template <typename T>
        void fill_typed(void* dst, const void* value, std::size_t elements) {
            auto* out = static_cast<T*>(dst);
            const auto typed_value = *static_cast<const T*>(value);

            for (std::size_t i = 0; i < elements; ++i) {
                out[i] = typed_value;
            }
        }
    }

    void fill_scalar(
        [[maybe_unused]] core::Device device,
        void* dst,
        const void* value,
        std::size_t elements,
        core::DType dtype
    ) {
        switch (dtype) {
            case core::DType::float64:
                fill_typed<double>(dst, value, elements);
                return;
            case core::DType::float32:
                fill_typed<float>(dst, value, elements);
                return;
            case core::DType::float16:
                fill_typed<std::uint16_t>(dst, value, elements);
                return;
            case core::DType::int32:
                fill_typed<std::int32_t>(dst, value, elements);
                return;
            case core::DType::boolean:
                fill_typed<bool>(dst, value, elements);
                return;
        }

        ORMI_THROW(DTypeError, "cpu backend cannot fill unknown dtype.");
    }
}
