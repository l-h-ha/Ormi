#include <cstdint>
#include <limits>
#include <utility>

#include <ormi/core/error.h>
#include <ormi/core/memory/ops.h>
#include <ormi/core/tensor.h>

#include "core/runtime/schedule.h"

namespace ormi::core {
    namespace {
        void require_copy_destination(const Tensor& tensor) {
            if (tensor.op() || !tensor.realized()) {
                ORMI_THROW(OperationError, "Copy destination must be a realized leaf tensor.");
            }
            if (!tensor.is_contiguous()) {
                ORMI_THROW(OperationError, "Copy destination must be contiguous.");
            }
        }

        bool overlaps(const void* lhs, const void* rhs, std::size_t bytes) {
            const auto left = reinterpret_cast<std::uintptr_t>(lhs);
            const auto right = reinterpret_cast<std::uintptr_t>(rhs);
            return left < right ? right - left < bytes : left - right < bytes;
        }

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

        void validate_strides(const Shape& shape, std::span<const std::int64_t> strides) {
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
          offset_bytes_(offset_bytes) {
        validate_strides(shape_, strides_);
        validate_storage_bounds(*buffer_, shape_, dtype_, strides_, offset_bytes_);
    }

    Tensor::Tensor(
        OpRef op,
        std::vector<std::shared_ptr<Tensor>> parents,
        Device device,
        Shape shape,
        DType dtype
    )
        : Tensor(op, std::monostate{}, std::move(parents), device, std::move(shape), dtype) {}

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
          offset_bytes_(0) {
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

            if (parent->device() != device_) {
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
          offset_bytes_(offset_bytes) {
        if (op_->output != Output::alias || op_->parents != 1 || op_->compute) {
            ORMI_THROW(OperationError, "An alias tensor requires a unary alias operation without compute.");
        }

        validate_strides(shape_, strides_);
    }

    Tensor::Tensor(OpRef op, std::vector<std::shared_ptr<Tensor>> parents)
        : op_(op.get()),
          attrs_(std::monostate{}),
          parents_(std::move(parents)),
          buffer_(nullptr),
          device_(require_parent(parents_.empty() ? nullptr : parents_[0])->device()),
          shape_(parents_[0]->shape_),
          dtype_(parents_[0]->dtype_),
          strides_(parents_[0]->strides_),
          offset_bytes_(parents_[0]->offset_bytes_) {
        if (op_->output != Output::alias || op_->prepare || parents_.empty() ||
            parents_.size() != op_->parents) {
            ORMI_THROW(OperationError, "An alias requires an alias operation and its declared parents.");
        }
        for (const auto& parent : parents_) {
            if (!parent) {
                ORMI_THROW(OperationError, "An alias cannot have a null parent.");
            }
            if (parent->device() != parents_[0]->device()) {
                ORMI_THROW(DeviceError, "An alias and its dependencies must use the same device.");
            }
        }
    }

    void Tensor::realize() {
        runtime::schedule::run(*this);
    }

    bool Tensor::realized() const {
        return static_cast<bool>(buffer_);
    }

    void Tensor::copy_from(const Tensor& source) {
        require_copy_destination(*this);
        if (shape_ != source.shape_) {
            ORMI_THROW(ShapeError, "Copy source and destination shapes must match.");
        }
        if (dtype_ != source.dtype_) {
            ORMI_THROW(DTypeError, "Copy source and destination dtypes must match.");
        }
        if (!source.realized()) {
            ORMI_THROW(OperationError, "Copy source must be realized before copying.");
        }
        if (!source.is_contiguous()) {
            ORMI_THROW(OperationError, "Copy source must be contiguous.");
        }

        const auto size = bytes();
        if (size == 0) {
            return;
        }
        auto* destination = data();
        const auto* value = source.data();
        if (device_ == source.device_ && destination == value) {
            return;
        }
        if (buffer_ == source.buffer_ && overlaps(destination, value, size)) {
            ORMI_THROW(OperationError, "Copy source and destination must not overlap.");
        }
        memory::copy(destination, device_, value, source.device_, size);
    }

    void Tensor::copy_from(std::span<const std::byte> source) {
        require_copy_destination(*this);
        if (source.size() != bytes()) {
            ORMI_THROW(ShapeError, "Host copy byte count must match the destination tensor.");
        }
        if (source.empty()) {
            return;
        }
        if (!source.data()) {
            ORMI_THROW(MemoryError, "Host copy source cannot be null.");
        }

        auto* destination = data();
        if (device_.is_host()) {
            if (destination == source.data()) {
                return;
            }
            if (overlaps(destination, source.data(), source.size())) {
                ORMI_THROW(OperationError, "Copy source and destination must not overlap.");
            }
        }
        memory::copy(destination, device_, source.data(), Device::cpu(), source.size());
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
