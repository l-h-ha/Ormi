#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include <ormi/core/ops/kind.h>
#include <ormi/core/runtime/backend_id.h>

namespace ormi::core {
    enum class DType;
    class Tensor;
    class Device;

    namespace memory {
        class Allocator;
    }
}

namespace ormi::core::runtime::fusion {
    struct Expr;
}

namespace ormi::core::runtime {
    inline constexpr std::uint32_t backend_abi = 1;

    using AllocatorFn = memory::Allocator* (*)(Device);
    using SyncFn = void (*)(Device);

    using CopyFn = void (*)(
        Device dst_device,
        void* dst,
        Device src_device,
        const void* src,
        std::size_t bytes
    );
    using FillFn = void (*)(Device device, void* dst, unsigned char value, std::size_t bytes);
    using FillScalarFn = void (*)(
        Device device,
        void* dst,
        const void* value,
        std::size_t elements,
        DType dtype
    );

    using PointwiseBinaryFn = void (*)(
        ops::BinaryOp op,
        Tensor& out,
        const Tensor& lhs,
        const Tensor& rhs
    );
    using PointwiseUnaryFn = void (*)(
        ops::UnaryOp op,
        Tensor& out,
        const Tensor& t
    );
    using CastFn = void (*)(Tensor& out, const Tensor& input);
    using SGDFn = void (*)(
        Tensor& out,
        const Tensor& parameter,
        const Tensor& gradient,
        double learning_rate
    );

    using ReduceFn = void (*)(
        ops::ReduceOp op,
        Tensor& out,
        const Tensor& input,
        std::span<const std::int64_t> axes,
        bool keepdims
    );
    using FiniteFn = bool (*)(std::span<const Tensor* const> inputs);
    using MatmulFn = void (*)(Tensor& out, const Tensor& lhs, const Tensor& rhs);
    using FusionFn = void (*)(Tensor& out, const fusion::Expr& expression);

    struct Backend {
        std::uint32_t abi;
        BackendId id;
        std::string_view name;

        AllocatorFn allocator;
        SyncFn synchronize;

        CopyFn copy;
        FillFn fill;
        FillScalarFn fill_scalar;

        CastFn cast;
        SGDFn sgd;
        MatmulFn matmul;
        ReduceFn reduce;
        FiniteFn all_finite;
        PointwiseBinaryFn pointwise_binary;
        PointwiseUnaryFn pointwise_unary;
        FusionFn fusion;
    };
}
