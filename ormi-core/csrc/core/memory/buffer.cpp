#include <utility>

#include <ormi/core/error.h>
#include <ormi/core/memory/allocator.h>
#include <ormi/core/memory/buffer.h>
#include <ormi/core/runtime/dispatch/device.h>

namespace ormi::core::memory {
    Buffer::Buffer(Device device, std::size_t bytes)
        : device_(device),
          bytes_(bytes),
          allocator_(runtime::device::allocator(device_))
    {
        if (!allocator_) {
            ORMI_THROW(MemoryError, "Device has no allocator.");
        }

        if (bytes_ != 0) {
            data_ = allocator_->allocate(bytes_);
        }
    }

    Buffer::~Buffer() {
        release();
    }

    Buffer::Buffer(Buffer&& other) noexcept
        : device_(other.device_),
          bytes_(std::exchange(other.bytes_, 0)),
          allocator_(std::exchange(other.allocator_, nullptr)),
          data_(std::exchange(other.data_, nullptr)) {}

    Buffer& Buffer::operator=(Buffer&& other) noexcept {
        if (this != &other) {
            release();

            device_ = other.device_;
            bytes_ = std::exchange(other.bytes_, 0);
            allocator_ = std::exchange(other.allocator_, nullptr);
            data_ = std::exchange(other.data_, nullptr);
        }

        return *this;
    }

    void* Buffer::data() const {
        return data_;
    }

    std::size_t Buffer::bytes() const {
        return bytes_;
    }

    Device Buffer::device() const {
        return device_;
    }

    Allocator* Buffer::allocator() const {
        return allocator_;
    }

    void Buffer::release() noexcept {
        if (allocator_ && data_) {
            allocator_->deallocate(data_, bytes_);
        }

        data_ = nullptr;
        bytes_ = 0;
        allocator_ = nullptr;
    }
}
