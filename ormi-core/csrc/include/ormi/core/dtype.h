#pragma once
#include <cstddef>
#include <ostream>
#include <string_view>

namespace ormi::core {
    enum class DType {
        float32,
        float16,
        int32,
        boolean,
        float64
    };

    constexpr bool is_floating(DType dtype) noexcept {
        return
            dtype == DType::float64 ||
            dtype == DType::float32 ||
            dtype == DType::float16;
    }

    constexpr bool is_integer(DType dtype) noexcept {
        return dtype == DType::int32;
    }

    constexpr bool is_boolean(DType dtype) noexcept {
        return dtype == DType::boolean;
    }

    constexpr bool is_numeric(DType dtype) noexcept {
        return is_floating(dtype) || is_integer(dtype);
    }

    std::string_view to_string(DType dtype);
    std::size_t dtype_size(DType dtype);
    DType promote(DType lhs, DType rhs);
    std::ostream& operator<<(std::ostream& os, DType dtype);
}
