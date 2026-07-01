#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <utility>
#include <cstdint>

#include "buffer.h"
#include "allocator.h"
#include "dtype.h"
#include "float16.h"

namespace ormi {
    Buffer::Buffer(void* p, size_t b, std::vector<int> s, Device d, DType dt) 
        : ptr(p), bytes(b), shape(std::move(s)), device(d), dtype(dt) {}
    
    Buffer::~Buffer() {
        if (ptr) {
            IAllocator* allocator = get_allocator(this->device);
            allocator->free(ptr, this->bytes);
        }
    }

    float Buffer::item(size_t index) const {
        size_t elements = this->bytes / dtype_size(this->dtype);
        if (index >= elements) {
            ORMI_THROW(BufferError, "Index out of bounds in Buffer::item()");
        }

        IAllocator* allocator = get_allocator(this->device);
        
        if (this->dtype == DType::Float32) {
            float val = 0.0f;
            const void* src_ptr = static_cast<const float*>(this->ptr) + index;
            allocator->copy_device_to_host(&val, src_ptr, sizeof(float));
            return val;
        } else if (this->dtype == DType::Float16) {
            std::uint16_t raw_bits = 0;
            const void* src_ptr = static_cast<const std::uint16_t*>(this->ptr) + index;
            allocator->copy_device_to_host(&raw_bits, src_ptr, sizeof(std::uint16_t));
            return fp16::bits_to_float(raw_bits);
        } else {
            ORMI_THROW(DTypeError, "item() does not support " + to_string(this->dtype) + ".");
        }
    }

    size_t Buffer::numel() const {
        size_t n = 1;

        for (int dim : shape) {
            n *= static_cast<size_t>(dim);
        }

        return n;
    }

    static void format_tensor(
        std::ostringstream& oss,
        const float* data,
        const std::vector<int>& shape,
        const std::vector<size_t>& strides,
        int dim,
        size_t offset,
        int edge_items = 5,
        int max_items_per_dim = 12
    ) {
        // Scalar
        if (shape.empty()) {
            oss << data[offset];
            return;
        }

        int dim_size = shape[dim];
        size_t stride = strides[dim];

        auto should_truncate = dim_size > max_items_per_dim;
        int front_count = should_truncate ? edge_items : dim_size;
        int back_start = should_truncate ? dim_size - edge_items : dim_size;

        // Vector
        if (dim == (int)shape.size() - 1) {
            oss << "[";

            for (int i = 0; i < front_count; ++i) {
                oss << data[offset + i * stride];
                if (i < front_count - 1) oss << ", ";
            }

            if (should_truncate) {
                oss << ", ..., ";
                for (int i = back_start; i < dim_size; ++i) {
                    oss << data[offset + i * stride];
                    if (i < dim_size - 1) oss << ", ";
                }
            }

            oss << "]";
            return;
        }

        // Matrix or tensor
        oss << "[";
        std::string indent(dim + 1, ' ');

        for (int i = 0; i < front_count; ++i) {
            format_tensor(
                oss,
                data,
                shape,
                strides,
                dim + 1,
                offset + i * stride,
                edge_items,
                max_items_per_dim
            );

            if (i < front_count - 1) {
                oss << ",\n" << indent;
            }
        }

        if (should_truncate) {
            oss << ",\n" << indent << "...,\n" << indent;

            for (int i = back_start; i < dim_size; ++i) {
                format_tensor(
                    oss,
                    data,
                    shape,
                    strides,
                    dim + 1,
                    offset + i * stride,
                    edge_items,
                    max_items_per_dim
                );

                if (i < dim_size - 1) {
                    oss << ",\n" << indent;
                }
            }
        }

        oss << "]";
    }

    std::string to_string(const Buffer& buffer) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(4);

        // Scalar
        if (buffer.shape.empty()) {
            oss << buffer.item();
            return oss.str();
        }

        // Strides
        std::vector<size_t> strides(buffer.shape.size(), 1);
        for (int i = (int)buffer.shape.size() - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * buffer.shape[i + 1];
        }

        // Transfer
        size_t elements = buffer.bytes / sizeof(float);
        std::vector<float> host_data(elements);
        IAllocator* allocator = get_allocator(buffer.device);

        allocator->copy_device_to_host(host_data.data(), buffer.ptr, buffer.bytes);
        
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

    //
    //
    //
    
    std::shared_ptr<Buffer> empty(const std::vector<int>& shape, Device device, DType dtype) {
        size_t elements = 1;
        for (int dim : shape) elements *= dim;
        
        size_t bytes = elements * dtype_size(dtype);
        
        IAllocator* allocator = get_allocator(device);
        void* ptr = allocator->allocate(bytes);
    
        return std::make_shared<Buffer>(ptr, bytes, shape, device, dtype);
    }

    std::shared_ptr<Buffer> full(const std::vector<int>& shape, float fill_value, Device device, DType dtype) {
        std::shared_ptr<Buffer> empty_buffer = empty(shape, device, dtype);

        size_t elements = 1;
        for (int dim : shape) elements *= dim;

        IAllocator* allocator = get_allocator(device);
        allocator->fill_n(empty_buffer->ptr, &fill_value, elements, dtype);

        return empty_buffer;
    }

    std::shared_ptr<Buffer> scalar(float value, Device device, DType dtype) {
        return full({}, value, device, dtype);
    }

    std::shared_ptr<Buffer> zeros(const std::vector<int>& shape, Device device, DType dtype) {
        return full(shape, 0.0f, device, dtype);
    }

    std::shared_ptr<Buffer> ones(const std::vector<int>& shape, Device device, DType dtype) {
        return full(shape, 1.0f, device, dtype);
    }

    std::shared_ptr<Buffer> zeros_like(const std::shared_ptr<Buffer>& array, DType dtype) {
        return full(array->shape, 0.0f, array->device, dtype);
    }

    std::shared_ptr<Buffer> zeros_like(const std::shared_ptr<Buffer>& array, Device device, DType dtype) {
        return full(array->shape, 0.0f, device, dtype);
    }

    std::shared_ptr<Buffer> ones_like(const std::shared_ptr<Buffer>& array, DType dtype) {
        return full(array->shape, 1.0f, array->device, dtype);
    }

    std::shared_ptr<Buffer> ones_like(const std::shared_ptr<Buffer>& array, Device device, DType dtype) {
        return full(array->shape, 1.0f, device, dtype);
    }
}
