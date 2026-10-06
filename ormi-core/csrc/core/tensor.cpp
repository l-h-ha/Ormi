#include <limits>
#include <utility>

#include <ormi/core/error.h>
#include <ormi/core/tensor.h>

#include "core/runtime/realize.h"

namespace ormi::core {
    namespace {
        std::size_t checked_nbytes(const Shape& shape, DType dtype) {
            const auto elements = shape.numel();
            const auto item_size = dtype_size(dtype);

            if (item_size != 0 && elements > std::numeric_limits<std::size_t>::max() / item_size) {
                ORMI_THROW(ShapeError, "Tensor byte size overflow.");
            }

            return elements * item_size;
        }

        const memory::Buffer& require_buffer(const std::shared_ptr<memory::Buffer>& buffer) {
            if (!buffer) {
                ORMI_THROW(BufferError, "Tensor requires a buffer.");
            }

            return *buffer;
        }

        std::shared_ptr<Tensor> require_parent(std::shared_ptr<Tensor> parent) {
            if (!parent) {
                ORMI_THROW(OperationError, "An aliasing operation requires a parent.");
            }

            return parent;
        }

        void strides_not_neg(const Shape& shape, std::span<const std::int64_t> strides) {
            if (shape.rank() != strides.size()) {
                ORMI_THROW(ShapeError, "Tensor stride rank must match shape rank.");
            }

            for (const auto stride : strides) {
                if (stride < 0) {
                    ORMI_THROW(ShapeError, "Tensor strides must be non-negative.");
                }
            }
        }

        std::size_t max_storage_offset(const Shape& shape, std::span<const std::int64_t> strides) {
            std::size_t offset = 0;

            for (std::size_t i = 0; i < shape.rank(); ++i) {
                if (shape[i] == 0) {
                    return 0;
                }

                const auto dim_offset = static_cast<std::size_t>(shape[i] - 1);
                const auto stride = static_cast<std::size_t>(strides[i]);

                if (stride != 0 && dim_offset > (std::numeric_limits<std::size_t>::max() - offset) / stride) {
                    ORMI_THROW(ShapeError, "Tensor view storage offset overflow.");
                }

                offset += dim_offset * stride;
            }

            return offset;
        }

        void validate_storage_bounds(
            const memory::Buffer& buffer,
            const Shape& shape,
            DType dtype,
            std::span<const std::int64_t> strides,
            std::size_t offset_bytes
        ) {
            if (offset_bytes > buffer.bytes()) {
                ORMI_THROW(BufferError, "Tensor view offset exceeds buffer size.");
            }

            if (shape.numel() == 0) {
                return;
            }

            const auto item_size = dtype_size(dtype);
            const auto max_offset = max_storage_offset(shape, strides);

            if (max_offset > (std::numeric_limits<std::size_t>::max() / item_size) - 1) {
                ORMI_THROW(ShapeError, "Tensor view byte offset overflow.");
            }

            const auto required_bytes = (max_offset + 1) * item_size;

            if (required_bytes > buffer.bytes() - offset_bytes) {
                ORMI_THROW(BufferError, "Tensor view exceeds buffer size.");
            }
        }
    }

    Tensor::Tensor(Device device, Shape shape, DType dtype)
        : op_(nullptr),
          attrs_(std::monostate{}),
          parents_(),
          buffer_(std::make_shared<memory::Buffer>(device, checked_nbytes(shape, dtype))),
          device_(device),
          shape_(std::move(shape)),
          dtype_(dtype),
          strides_(contiguous_strides(shape_)),
          offset_bytes_(0) {}

    Tensor::Tensor(
        std::shared_ptr<memory::Buffer> buffer,
        Shape shape,
        DType dtype,
        std::vector<std::int64_t> strides,
        std::size_t offset_bytes
    )
        : op_(nullptr),
          attrs_(std::monostate{}),
          parents_(),
          buffer_(std::move(buffer)),
          device_(require_buffer(buffer_).device()),
          shape_(std::move(shape)),
          dtype_(dtype),
          strides_(strides.empty() ? contiguous_strides(shape_) : std::move(strides)),
          offset_bytes_(offset_bytes)
    {
        strides_not_neg(shape_, strides_);
        validate_storage_bounds(*buffer_, shape_, dtype_, strides_, offset_bytes_);
    }

    Tensor::Tensor(
        OpRef op,
        std::vector<std::shared_ptr<Tensor>> parents,
        Device device,
        Shape shape,
        DType dtype
    )
        : Tensor(
              op,
              std::monostate{},
              std::move(parents),
              device,
              std::move(shape),
              dtype
          ) {}

    Tensor::Tensor(
        OpRef op,
        Attrs attrs,
        std::vector<std::shared_ptr<Tensor>> parents,
        Device device,
        Shape shape,
        DType dtype
    )
        : op_(op.get()),
          attrs_(std::move(attrs)),
          parents_(std::move(parents)),
          buffer_(nullptr),
          device_(device),
          shape_(std::move(shape)),
          dtype_(dtype),
          strides_(contiguous_strides(shape_)),
          offset_bytes_(0)
    {
        if (parents_.size() != op_->parents) {
            ORMI_THROW(OperationError, "Lazy tensor parent count does not match its operation.");
        }

        if (op_->output != Output::allocate) {
            ORMI_THROW(OperationError, "A compute tensor requires an allocating operation.");
        }

        if (!op_->compute) {
            ORMI_THROW(OperationError, "An allocating operation requires a compute function.");
        }

        for (const auto& parent : parents_) {
            if (!parent) {
                ORMI_THROW(OperationError, "A lazy tensor cannot have a null parent.");
            }

            if (
                parent->device().backend_id() != device_.backend_id() ||
                parent->device().index() != device_.index()
            ) {
                ORMI_THROW(DeviceError, "A lazy tensor and its parents must use the same device.");
            }

        }
    }

    Tensor::Tensor(
        OpRef op,
        Attrs attrs,
        std::shared_ptr<Tensor> parent,
        Shape shape,
        std::vector<std::int64_t> strides,
        std::size_t offset_bytes
    )
        : op_(op.get()),
          attrs_(std::move(attrs)),
          parents_{require_parent(std::move(parent))},
          buffer_(nullptr),
          device_(parents_[0]->device()),
          shape_(std::move(shape)),
          dtype_(parents_[0]->dtype()),
          strides_(std::move(strides)),
          offset_bytes_(offset_bytes)
    {
        if (op_->output != Output::alias || op_->parents != 1 || op_->compute) {
            ORMI_THROW(OperationError, "An alias tensor requires a unary alias operation without compute.");
        }

        strides_not_neg(shape_, strides_);
    }

    void Tensor::realize() {
        runtime::realize(*this);
    }

    bool Tensor::realized() const {
        return static_cast<bool>(buffer_);
    }

    void* Tensor::data() {
        realize();
        auto* bytes = static_cast<unsigned char*>(buffer_->data());
        return bytes ? bytes + offset_bytes_ : nullptr;
    }

    const void* Tensor::data() const {
        const_cast<Tensor*>(this)->realize();
        const auto* bytes = static_cast<const unsigned char*>(buffer_->data());
        return bytes ? bytes + offset_bytes_ : nullptr;
    }

    std::shared_ptr<memory::Buffer> Tensor::buffer() const {
        const_cast<Tensor*>(this)->realize();
        return buffer_;
    }

    const Op* Tensor::op() const {
        return op_;
    }

    const Attrs& Tensor::attrs() const {
        return attrs_;
    }

    const std::vector<std::shared_ptr<Tensor>>& Tensor::parents() const {
        return parents_;
    }

    Device Tensor::device() const {
        return device_;
    }

    DType Tensor::dtype() const {
        return dtype_;
    }

    const Shape& Tensor::shape() const {
        return shape_;
    }

    std::span<const std::int64_t> Tensor::strides() const {
        return strides_;
    }

    std::size_t Tensor::offset_bytes() const {
        return offset_bytes_;
    }

    std::size_t Tensor::bytes() const {
        return checked_nbytes(shape_, dtype_);
    }

    std::size_t Tensor::numel() const {
        return shape_.numel();
    }

    std::size_t Tensor::rank() const {
        return shape_.rank();
    }

    bool Tensor::is_contiguous() const {
        return ormi::core::is_contiguous(shape_, strides_);
    }
}
