#include <stdexcept>
#include <sstream>
#include <iomanip>
#include "memory.h"
#include "allocator.h"
#include "core/dtype.h"

namespace ormi {
    Buffer::Buffer(void* p, size_t b, std::vector<int> s, Device d) 
        : ptr(p), bytes(b), shape(s), device(d) {}
    
    Buffer::~Buffer() {
        if (ptr) {
            IAllocator* allocator = get_allocator(this->device.type);
            allocator->free(ptr, this->bytes);
        }
    }

    static void format_tensor(std::ostringstream& oss, const float* data, 
                              const std::vector<int>& shape, const std::vector<size_t>& strides, 
                              int dim, size_t offset) {
        // Scalar
        if (shape.empty()) {
            oss << data[offset];
            return;
        }

        // Vector
        if (dim == shape.size() - 1) {
            oss << "[";
            for (int i = 0; i < shape[dim]; ++i) {
                oss << data[offset + i * strides[dim]];
                if (i < shape[dim] - 1) oss << ", ";
            }
            oss << "]";
            return;
        }

        // Matrix or tensor
        oss << "[";
        for (int i = 0; i < shape[dim]; ++i) {
            format_tensor(oss, data, shape, strides, dim + 1, offset + i * strides[dim]);
            if (i < shape[dim] - 1) {
                oss << ",\n" << std::string(dim + 1, ' '); 
            }
        }
        oss << "]";
    }

    std::string to_string(const Buffer& buffer) {
        // Strides
        std::vector<size_t> strides(buffer.shape.size(), 1);
        if (!buffer.shape.empty()) {
            for (int i = (int)buffer.shape.size() - 2; i >= 0; --i) {
                strides[i] = strides[i + 1] * buffer.shape[i + 1];
            }
        }

        // Transfer
        size_t elements = buffer.bytes / sizeof(float);
        std::vector<float> host_data(elements);
        IAllocator* allocator = get_allocator(buffer.device.type);

        allocator->copy_device_to_host(host_data.data(), buffer.ptr, buffer.bytes);
        
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(4);
        
        format_tensor(oss, host_data.data(), buffer.shape, strides, 0, 0);
        return oss.str();
    }

    std::string to_string(const std::shared_ptr<Buffer>& buffer) {
        if (!buffer) return "null";
        return to_string(*buffer);
    }

    std::ostream& operator<<(std::ostream& os, const Buffer& buffer) {
        os << to_string(buffer);
        return os;
    }

    ///
    ///
    ///
    
    std::shared_ptr<Buffer> empty(std::vector<int> shape, DeviceEnum dev_type) {
        size_t bytes = sizeof(float);
        for (int dim : shape) bytes *= dim;
        
        IAllocator* allocator = get_allocator(dev_type);
        void* ptr = allocator->allocate(bytes);
    
        return std::make_shared<Buffer>(ptr, bytes, shape, Device(dev_type));
    }

    std::shared_ptr<Buffer> full(std::vector<int> shape, float fill_value, DeviceEnum dev_type) {
        std::shared_ptr<Buffer> empty_buffer = empty(shape, dev_type);

        size_t elements = 1;
        for (int dim : shape) elements *= dim;

        IAllocator* allocator = get_allocator(dev_type);
        allocator->fill_n(empty_buffer->ptr, &fill_value, elements, DType::Float32);
        
        return empty_buffer;
    }

    ///
    ///
    ///

    std::shared_ptr<Buffer> scalar(float val, DeviceEnum dev_type) {
        return full({1}, val, dev_type);
    }

    std::shared_ptr<Buffer> zeros(std::vector<int> shape, DeviceEnum dev_type) {
        return full(shape, 0.0f, dev_type);
    }

    std::shared_ptr<Buffer> ones(std::vector<int> shape, DeviceEnum dev_type) {
        return full(shape, 1.0f, dev_type);
    }

    std::shared_ptr<Buffer> zeros_like(std::shared_ptr<Buffer> array) {
        return full(array->shape, 0.0f, array->device.type);
    }

    std::shared_ptr<Buffer> zeros_like(std::shared_ptr<Buffer> array, DeviceEnum dev_type) {
        return full(array->shape, 0.0f, dev_type);
    }

    std::shared_ptr<Buffer> ones_like(std::shared_ptr<Buffer> array) {
        return full(array->shape, 1.0f, array->device.type);
    }

    std::shared_ptr<Buffer> ones_like(std::shared_ptr<Buffer> array, DeviceEnum dev_type) {
        return full(array->shape, 1.0f, dev_type);
    }
}