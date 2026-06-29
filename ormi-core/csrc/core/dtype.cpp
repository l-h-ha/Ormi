#include <algorithm>
#include <cctype>
#include <magic_enum.hpp>

#include "core/dtype.h"
#include "core/error.h"

namespace ormi {
    std::string to_string(DType type) {
        auto name = magic_enum::enum_name(type);
        if (name.empty()) {
            ORMI_THROW(DTypeError, "Unrecognized DType");
        }

        // ex. "Float32" -> "float32"
        std::string t(name);
        std::transform(t.begin(), t.end(), t.begin(), 
                       [](unsigned char c){ return std::tolower(c); });
        
        return t;
    }
    
    size_t dtype_size(DType type) {
        switch (type) {
            case DType::Float32: return 4;
            case DType::Float16: return 2;
            case DType::Int32:   return 4;
            case DType::Bool:    return 1;
            default: ORMI_THROW(DTypeError, "Unknown DType size mapping requested.");
        }
    }

    std::ostream& operator<<(std::ostream& os, DType type) {
        return os << to_string(type);
    }
}