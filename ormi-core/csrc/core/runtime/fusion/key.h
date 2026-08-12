#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <ormi/backend/fusion.h>

namespace ormi::core::runtime::fusion {
    class Key final {
    public:
        explicit Key(const Expr& expression);

        bool matches(const Expr& expression) const;
        std::uint64_t hash() const noexcept;
        static std::uint64_t hash(const Expr& expression);

    private:
        DType dtype_;
        std::vector<std::size_t> shape_;
        std::vector<Node> nodes_;
        Index output_;
        std::vector<Index> auxiliary_outputs_;
        std::optional<Reduction> reduction_;
        std::uint64_t hash_;
    };
}
