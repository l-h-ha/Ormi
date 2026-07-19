#pragma once
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <ostream>
#include <span>
#include <vector>

namespace ormi::core {
    class Shape {
    public:
        Shape() = default;
        Shape(std::initializer_list<std::int64_t> dims);
        explicit Shape(std::vector<std::int64_t> dims);

        std::size_t rank() const;
        std::size_t numel() const;

        bool scalar() const;
        bool empty() const;

        std::int64_t operator[](std::size_t index) const;
        const std::vector<std::int64_t>& dims() const;
        std::span<const std::int64_t> span() const;

        friend bool operator==(const Shape&, const Shape&) = default;

    private:
        std::vector<std::int64_t> dims_;
    };

    std::vector<std::int64_t> contiguous_strides(const Shape& shape);
    bool is_contiguous(const Shape& shape, std::span<const std::int64_t> strides);
    Shape broadcast(const Shape& lhs, const Shape& rhs);

    std::ostream& operator<<(std::ostream& os, const Shape& shape);
}
