#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "core/memory.h"
#include "core/allocator.h"

namespace py = pybind11;

namespace ormi {
    std::shared_ptr<Buffer> numpy_to_buffer(py::array_t<float> array, DeviceEnum dev_type) {
        py::buffer_info buf = array.request();
        size_t bytes = buf.size * sizeof(float);
    
        std::vector<int> shape;
        shape.reserve(buf.shape.size());
        for (auto dim : buf.shape) {
            shape.push_back(static_cast<int>(dim));
        }
        
        IAllocator* allocator = get_allocator(dev_type);
        void* ptr = allocator->allocate(bytes);
        allocator->copy_host_to_device(ptr, buf.ptr, bytes);
    
        return std::make_shared<Buffer>(ptr, bytes, shape, Device(dev_type));
    }
    
    py::array_t<float> buffer_to_numpy(const std::shared_ptr<Buffer>& buffer) {
        if (!buffer || !buffer->ptr) {
            throw std::runtime_error("Cannot read from an empty or unrealized memory buffer.");
        }
    
        IAllocator* allocator = get_allocator(buffer->device.type);
    
        if (allocator->is_host_visible()) {
            py::capsule keep_alive(new std::shared_ptr<ormi::Buffer>(buffer), [](void* p) {
                delete static_cast<std::shared_ptr<ormi::Buffer>*>(p);
            });
            
            std::vector<py::ssize_t> strides(buffer->shape.size());
            py::ssize_t stride = sizeof(float);
            for (int i = (int)buffer->shape.size() - 1; i >= 0; --i) {
                strides[i] = stride;
                stride *= buffer->shape[i];
            }
    
            return py::array_t<float>(
                buffer->shape,
                strides,
                static_cast<float*>(buffer->ptr),
                keep_alive
            );
        } else {
            auto result = py::array_t<float>(buffer->shape);
            py::buffer_info buf = result.request();
            
            allocator->copy_device_to_host(buf.ptr, buffer->ptr, buffer->bytes);
            
            return result;
        }
    }

    void bind_memory(py::module_& m) {
        py::class_<ormi::Buffer, std::shared_ptr<ormi::Buffer>>(m, "DataBuffer")
        .def_property_readonly("bytes", [](const ormi::Buffer& b) { return b.bytes; })
        .def_property_readonly("shape", [](const ormi::Buffer& b) { return b.shape; });
        
        m.def("numpy_to_buffer", &ormi::numpy_to_buffer);
        m.def("buffer_to_numpy", &ormi::buffer_to_numpy);
    }
}