#pragma once
#include <cstddef>

#include <ormi/core/device.h>

namespace ormi::core::memory {
    class Allocator;

    // Owns bytes from one device allocator, with no dtype, shape, or view offset.
    // Tensor views share a Buffer. Destruction returns storage to the allocator,
    // which may keep it cached for reuse or free it to the OS or driver.
    class Buffer {
    public:
        Buffer(Device device, std::size_t bytes);
        ~Buffer();

        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;

        Buffer(Buffer&& other) noexcept;
        Buffer& operator=(Buffer&& other) noexcept;

        // Base allocation pointer, or null for a zero-byte or moved-from buffer.
        void* data() const;
        std::size_t bytes() const;
        Device device() const;
        Allocator* allocator() const;

    private:
        void release() noexcept;

        Device device_;
        std::size_t bytes_ = 0;
        Allocator* allocator_ = nullptr;
        void* data_ = nullptr;
    };
}
