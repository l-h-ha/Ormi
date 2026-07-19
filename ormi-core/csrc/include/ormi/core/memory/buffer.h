#pragma once
#include <cstddef>

#include <ormi/core/device.h>

namespace ormi::core::memory {
    class Allocator;

    class Buffer {
    public:
        Buffer(Device device, std::size_t bytes);
        ~Buffer();

        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;

        Buffer(Buffer&& other) noexcept;
        Buffer& operator=(Buffer&& other) noexcept;

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
