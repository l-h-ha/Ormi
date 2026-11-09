#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <ormi/core/attrs.h>
#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/memory/buffer.h>
#include <ormi/core/op.h>
#include <ormi/core/shape.h>

namespace ormi::core {
    class Plan;
}

namespace ormi::core::runtime {
    class Step;
}

namespace ormi::core {
    // Stores shape, dtype, input tensors, and optional shared storage.
    // Computing values keeps the operation and inputs, so gradients still reach them.
    class Tensor {
    public:
        // Allocate a contiguous input tensor on device. Its values are uninitialized.
        Tensor(Device device, Shape shape, DType dtype);
        // Alias parents[0]'s layout and storage after all parents have executed.
        Tensor(OpRef op, std::vector<std::shared_ptr<Tensor>> parents);
        // Share an existing allocation. Strides are in elements and offset_bytes is in bytes.
        Tensor(
            std::shared_ptr<memory::Buffer> buffer,
            Shape shape,
            DType dtype,
            std::vector<std::int64_t> strides = {},
            std::size_t offset_bytes = 0
        );
        // Build a lazy operation that allocates output storage when run. Shape and dtype are known.
        Tensor(
            OpRef op,
            std::vector<std::shared_ptr<Tensor>> parents,
            Device device,
            Shape shape,
            DType dtype
        );
        Tensor(
            OpRef op,
            Attrs attrs,
            std::vector<std::shared_ptr<Tensor>> parents,
            Device device,
            Shape shape,
            DType dtype
        );
        // Build a lazy view with a new shape/stride layout over its parent's allocation.
        Tensor(
            OpRef op,
            Attrs attrs,
            std::shared_ptr<Tensor> parent,
            Shape shape,
            std::vector<std::int64_t> strides,
            std::size_t offset_bytes
        );

        // Execute missing dependencies once. Use Plan to rerun the same graph.
        void realize();
        bool realized() const;

        // Replace data in an existing, contiguous leaf without changing its graph identity.
        // Tensor sources must be realized, contiguous, and match shape and dtype.
        // Copies finish before returning. Use Plan::run() to recompute existing outputs.
        void copy_from(const Tensor& source);
        // Host bytes in this tensor's dtype and logical order. Size must match bytes().
        void copy_from(std::span<const std::byte> source);

        // These accessors may realize the graph, including the const overloads.
        // data() includes the view offset. buffer() refers to the whole allocation.
        // A CUDA data pointer is not host-readable, and queued work may still be running.
        void* data();
        const void* data() const;
        std::shared_ptr<memory::Buffer> buffer() const;

        // Reading metadata does not run the graph. A null op means this is a leaf (no operation).
        const Op* op() const;
        const Attrs& attrs() const;
        const std::vector<std::shared_ptr<Tensor>>& parents() const;
        Device device() const;
        DType dtype() const;
        const Shape& shape() const;
        std::span<const std::int64_t> strides() const;

        std::size_t offset_bytes() const;
        std::size_t bytes() const;
        std::size_t numel() const;
        std::size_t rank() const;

        bool is_contiguous() const;

    private:
        friend class Plan;
        friend class runtime::Step;

        const Op* op_;
        Attrs attrs_;
        std::vector<std::shared_ptr<Tensor>> parents_;
        std::shared_ptr<memory::Buffer> buffer_;
        Device device_;
        Shape shape_;
        DType dtype_;
        std::vector<std::int64_t> strides_;
        std::size_t offset_bytes_;
    };
}
