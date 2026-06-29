#include "bindings/numpy_buffer.h"

#include <stdexcept>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/allocator.h"
#include "core/buffer.h"

namespace py = pybind11;

namespace ormi {
    std::shared_ptr<Buffer> numpy_to_buffer(
        py::array_t<float, py::array::c_style | py::array::forcecast> array,
        Device device
    ) {
        py::buffer_info buf = array.request();
        const size_t bytes = static_cast<size_t>(buf.size) * sizeof(float);

        std::vector<int> shape;
        shape.reserve(buf.shape.size());

        for (auto dim : buf.shape) {
            shape.push_back(static_cast<int>(dim));
        }

        IAllocator* allocator = get_allocator(device);
        void* ptr = allocator->allocate(bytes);
        allocator->copy_host_to_device(ptr, buf.ptr, bytes);

        return std::make_shared<Buffer>(
            ptr,
            bytes,
            shape,
            device,
            DType::Float32
        );
    }

    py::array_t<float> buffer_to_numpy(const std::shared_ptr<Buffer>& buffer) {
        if (!buffer || !buffer->ptr) {
            throw std::runtime_error("Cannot read from an empty or unrealized buffer.");
        }

        if (buffer->dtype != DType::Float32) {
            throw std::runtime_error("buffer_to_numpy currently only supports Float32 buffers.");
        }

        IAllocator* allocator = get_allocator(buffer->device);

        std::vector<py::ssize_t> shape;
        shape.reserve(buffer->shape.size());

        for (int dim : buffer->shape) {
            shape.push_back(static_cast<py::ssize_t>(dim));
        }

        if (allocator->is_host_visible()) {
            py::capsule keep_alive(new std::shared_ptr<Buffer>(buffer), [](void* p) {
                delete static_cast<std::shared_ptr<Buffer>*>(p);
            });

            std::vector<py::ssize_t> strides(shape.size());
            py::ssize_t stride = static_cast<py::ssize_t>(sizeof(float));

            for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
                strides[static_cast<size_t>(i)] = stride;
                stride *= shape[static_cast<size_t>(i)];
            }

            return py::array_t<float>(
                shape,
                strides,
                static_cast<float*>(buffer->ptr),
                keep_alive
            );
        }

        py::array_t<float> result(shape);
        py::buffer_info out = result.request();

        allocator->copy_device_to_host(out.ptr, buffer->ptr, buffer->bytes);

        return result;
    }
}