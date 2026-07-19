#include <algorithm>
#include <limits>
#include <sstream>

#include <ormi/core/error.h>
#include <ormi/core/shape.h>

namespace ormi::core {
    namespace {
        void dims_not_neg(std::span<const std::int64_t> dims) {
            for (const auto dim : dims) {
                if (dim < 0) {
                    ORMI_THROW(ShapeError, "Shape dimensions must be non-negative.");
                }
            }
        }
    }

    Shape::Shape(std::initializer_list<std::int64_t> dims)
        : dims_(dims) {
        dims_not_neg(dims_);
    }

    Shape::Shape(std::vector<std::int64_t> dims)
        : dims_(std::move(dims)) {
        dims_not_neg(dims_);
    }

    std::size_t Shape::rank() const {
        return dims_.size();
    }

    std::size_t Shape::numel() const {
        std::size_t result = 1;

        for (const auto dim : dims_) {
            const auto unsigned_dim = static_cast<std::size_t>(dim);

            if (unsigned_dim != 0 && result > std::numeric_limits<std::size_t>::max() / unsigned_dim) {
                ORMI_THROW(ShapeError, "Shape element count overflow.");
            }

            result *= unsigned_dim;
        }

        return result;
    }

    bool Shape::scalar() const {
        return dims_.empty();
    }

    bool Shape::empty() const {
        return numel() == 0;
    }

    std::int64_t Shape::operator[](std::size_t index) const {
        return dims_.at(index);
    }

    const std::vector<std::int64_t>& Shape::dims() const {
        return dims_;
    }

    std::span<const std::int64_t> Shape::span() const {
        return dims_;
    }

    std::vector<std::int64_t> contiguous_strides(const Shape& shape) {
        std::vector<std::int64_t> strides(shape.rank(), 1);

        for (std::size_t i = shape.rank(); i > 1; --i) {
            const auto current = i - 2;
            const auto next = i - 1;

            strides[current] = strides[next] * shape[next];
        }

        return strides;
    }

    bool is_contiguous(const Shape& shape, std::span<const std::int64_t> strides) {
        if (shape.rank() != strides.size()) {
            return false;
        }

        std::int64_t expected = 1;

        for (std::size_t i = shape.rank(); i > 0; --i) {
            const auto index = i - 1;

            if (strides[index] != expected) {
                return false;
            }

            expected *= shape[index];
        }

        return true;
    }

    Shape broadcast(const Shape& lhs, const Shape& rhs) {
        const auto rank = std::max(lhs.rank(), rhs.rank());
        std::vector<std::int64_t> dims(rank, 1);

        for (std::size_t out = 0; out < rank; ++out) {
            const auto lhs_offset = rank - lhs.rank();
            const auto rhs_offset = rank - rhs.rank();
            const auto lhs_dim = out < lhs_offset ? 1 : lhs[out - lhs_offset];
            const auto rhs_dim = out < rhs_offset ? 1 : rhs[out - rhs_offset];

            if (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1) {
                std::ostringstream oss;
                oss << "Shapes cannot broadcast: " << lhs << " and " << rhs << ".";
                ORMI_THROW(ShapeError, oss.str());
            }

            dims[out] = lhs_dim == 1 ? rhs_dim : lhs_dim;
        }

        return Shape(std::move(dims));
    }

    std::ostream& operator<<(std::ostream& os, const Shape& shape) {
        os << "[";

        for (std::size_t i = 0; i < shape.rank(); ++i) {
            os << shape[i];

            if (i + 1 != shape.rank()) {
                os << ", ";
            }
        }

        return os << "]";
    }
}
