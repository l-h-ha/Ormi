#include <ormi/core/dtype.h>

#include <ormi/core/error.h>

namespace ormi::core {
    std::string_view to_string(DType dtype) {
        switch (dtype) {
            case DType::float64:
                return "float64";
            case DType::float32:
                return "float32";
            case DType::float16:
                return "float16";
            case DType::int32:
                return "int32";
            case DType::boolean:
                return "bool";
        }

        ORMI_THROW(DTypeError, "Unknown dtype.");
    }

    std::size_t dtype_size(DType dtype) {
        switch (dtype) {
            case DType::float64:
                return 8;
            case DType::float32:
                return 4;
            case DType::float16:
                return 2;
            case DType::int32:
                return 4;
            case DType::boolean:
                return 1;
        }

        ORMI_THROW(DTypeError, "Unknown dtype size.");
    }

    DType promote(DType lhs, DType rhs) {
        if (!is_numeric(lhs) || !is_numeric(rhs)) {
            ORMI_THROW(DTypeError, "Dtype promotion requires numeric dtypes.");
        }

        if (lhs == rhs) {
            return lhs;
        }

        if (is_integer(lhs) || is_integer(rhs)) {
            ORMI_THROW(
                DTypeError,
                "Implicit promotion between integer and floating-point dtypes is not supported."
            );
        }

        if (lhs == DType::float64 || rhs == DType::float64) {
            return DType::float64;
        }

        return DType::float32;
    }

    std::ostream& operator<<(std::ostream& os, DType dtype) {
        return os << to_string(dtype);
    }
}
