#pragma once
#include <cstddef>

namespace ormi::core::memory {
    class Allocator {
    public:
        virtual ~Allocator() = default;

        virtual void* allocate(std::size_t bytes) = 0;
        virtual void deallocate(void* ptr, std::size_t bytes) noexcept = 0;
        virtual void empty_cache() = 0;
        virtual bool host_visible() const = 0;
    };
}
