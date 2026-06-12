#pragma once
#include <cstddef>
#include "core/device.h"
#include "core/dtype.h"

namespace ormi {
    class IAllocator {
    public:
        virtual ~IAllocator() = default;
        
        virtual void* allocate(size_t bytes) = 0;
        virtual void free(void* ptr, size_t bytes) = 0;
        virtual void memset(void* ptr, int value, size_t bytes) = 0;
        virtual void memcpy(void* dst, const void* src, size_t size) = 0;
        virtual void fill_n(void* dst, const void* val_ptr, size_t bytes, DType dtype) = 0;
        
        virtual void copy_device_to_host(void* dst, const void* src, size_t bytes) = 0;
        virtual void copy_host_to_device(void* dst, const void* src, size_t bytes) = 0;

        virtual bool is_host_visible() const = 0;
    };
    
    IAllocator* get_cpu_allocator();
    IAllocator* get_cuda_allocator();

    IAllocator* get_allocator(DeviceEnum type);

}