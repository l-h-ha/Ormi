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
    class Tensor {
    public:
        Tensor(Device device, Shape shape, DType dtype);
        Tensor(
            std::shared_ptr<memory::Buffer> buffer,
            Shape shape,
            DType dtype,
            std::vector<std::int64_t> strides = {},
            std::size_t offset_bytes = 0
        );
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
        Tensor(
            OpRef op,
            Attrs attrs,
            std::shared_ptr<Tensor> parent,
            Shape shape,
            std::vector<std::int64_t> strides,
            std::size_t offset_bytes
        );

        void realize();
        bool realized() const;

        void* data();
        const void* data() const;
        std::shared_ptr<memory::Buffer> buffer() const;

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
