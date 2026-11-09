#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

#include <ormi/backend/fusion.h>
#include <ormi/core/ops/kind.h>

namespace ormi::core {
    struct Caputo;
    struct Pick;
    struct Bernoulli;
    enum class DType;
    class Setup;
    class Tensor;
    class Device;

    namespace memory {
        class Allocator;
    }
}

namespace ormi::core::runtime {
    // Bump when the table layout or a callback signature changes. Plugins must match exactly.
    inline constexpr std::uint32_t backend_abi = 18;

    using AllocatorFn = memory::Allocator* (*)(Device);
    using SyncFn = void (*)(Device);

    enum class Transfer : std::uint8_t { within, from_host, to_host };

    using CopyFn = void (*)(
        Transfer transfer,
        Device dst_device,
        void* dst,
        Device src_device,
        const void* src,
        std::size_t bytes
    );
    using DirectCopyFn =
        bool (*)(Device dst_device, void* dst, Device src_device, const void* src, std::size_t bytes);
    // Same-device copy that may return before completion. Keep both allocations alive until it completes.
    using CopyAsyncFn = void (*)(Device device, void* dst, const void* src, std::size_t bytes);
    using FillFn = void (*)(Device device, void* dst, unsigned char value, std::size_t bytes);
    using FillScalarFn =
        void (*)(Device device, void* dst, const void* value, std::size_t elements, DType dtype);

    using PointwiseBinaryFn = void (*)(ops::BinaryOp op, Tensor& out, const Tensor& lhs, const Tensor& rhs);
    using PointwiseUnaryFn = void (*)(ops::UnaryOp op, Tensor& out, const Tensor& t);
    using CastFn = void (*)(Tensor& out, const Tensor& input);
    // Write an inverted-dropout mask: zero with probability p, otherwise 1/(1-p).
    // Output storage is Float32 or Float64. Float16 dropout computes its mask in Float32.
    // Advance the counter during execution, including device execution of a recording.
    using BernoulliFn = void (*)(Tensor& out, const Bernoulli& attrs);
    using SGDFn =
        void (*)(Tensor& out, const Tensor& parameter, const Tensor& gradient, double learning_rate);

    using ReduceFn = void (*)(
        ops::ReduceOp op,
        Tensor& out,
        const Tensor& input,
        std::span<const std::int64_t> axes,
        bool keepdims
    );
    using FiniteFn = bool (*)(std::span<const Tensor* const> inputs);
    using MatmulFn = void (*)(Tensor& out, const Tensor& lhs, const Tensor& rhs);
    using IndexFn = void (*)(Tensor& out, const Tensor& input, const Tensor& indices, const Pick& attrs);
    using ScatterScalarFn = void (*)(Tensor& out, const Tensor& scalar, std::size_t index);
    using FractionalFn = void (*)(
        ops::FractionalOp op,
        Tensor& out,
        const Tensor& input,
        const Tensor* initial_derivative,
        const Caputo& attrs
    );
    using FractionalPrepareFn = Setup (*)(ops::FractionalOp op, Tensor& out, const Caputo& attrs);
    using SupportsFusionFn = bool (*)(const fusion::Expr& expression);
    using PrepareFusionFn =
        Setup (*)(Tensor& out, std::span<Tensor* const> auxiliary_outputs, const fusion::Expr& expression);

    // The job, expression, output-pointer array, and setup slot are valid only during preparation.
    // Copy any preparation data needed later. The caller keeps Tensor objects alive through execution.
    struct FusionJob {
        Tensor* output;
        Tensor* const* auxiliary_outputs;
        std::size_t auxiliary_count;
        const fusion::Expr* expression;
        Setup* setup;
    };

    // Write through each setup slot. An empty Setup selects direct execution.
    // If preparation throws, installed states remain owned by their steps.
    // The batch stays pending and a retry may replace those states.
    using PrepareFusionsFn = void (*)(std::span<FusionJob> jobs);
    using FusionFn =
        void (*)(Tensor& out, std::span<Tensor* const> auxiliary_outputs, const fusion::Expr& expression);
    using RecordBodyFn = void (*)(void* context);
    using RecordFn = void* (*)(Device device, RecordBodyFn body, void* context);
    using ReplayFn = void (*)(void* recording);
    using ReleaseFn = void (*)(void* recording) noexcept;

    struct Memory {
        AllocatorFn allocator;
        SyncFn synchronize;
        CopyFn copy;
        DirectCopyFn direct_copy;
        CopyAsyncFn copy_async;
        FillFn fill;
        FillScalarFn fill_scalar;
    };

    // Inputs and output storage are ready before callbacks compute values.
    // Graph construction and gradient rules live in core/ops and nn.
    struct Ops {
        PointwiseBinaryFn binary;
        PointwiseUnaryFn unary;
        CastFn cast;
        MatmulFn matmul;
        ReduceFn reduce;
        FiniteFn all_finite;
        SGDFn sgd;
        IndexFn pick;
        IndexFn unpick;
        BernoulliFn bernoulli;
        // Optional: write the scalar at a logical index and zero every other element.
        // Core falls back to fill followed by copy_async when this callback is absent.
        ScatterScalarFn scatter_scalar;
    };

    struct Fractional {
        FractionalFn execute;
        FractionalPrepareFn prepare;
    };

    struct Fusion {
        SupportsFusionFn supports;
        PrepareFusionFn prepare;
        PrepareFusionsFn prepare_many;
        FusionFn execute;
    };

    // record calls the host body once. replay runs only the captured device work.
    // Capture any state updates that must run on every replay.
    struct Recording {
        RecordFn record;
        ReplayFn replay;
        ReleaseFn release;
    };

    // Required memory callbacks are checked at load time. direct_copy is optional.
    // normalize() fills missing Ops callbacks with functions that throw when called.
    // The optional scatter_scalar callback keeps its core fallback.
    struct Backend {
        std::uint32_t abi;
        Memory memory;
        Ops ops;
        Fractional fractional;
        Fusion fusion;
        Recording recording;
    };

    static_assert(std::is_standard_layout_v<Backend>);
    static_assert(std::is_trivially_copyable_v<Backend>);
}
