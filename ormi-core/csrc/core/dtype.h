#pragma once
#include <cstddef>
#include <ostream>
#include <string>

namespace ormi {
    enum class DType {
        Float32,
        Float16,
        Int32,
        Bool
    };

    std::string to_string(DType type);
    size_t dtype_size(DType type);
    std::ostream& operator<<(std::ostream& os, DType type);
}