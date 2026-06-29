#pragma once
#include <string>
#include <iostream>
#include <vector>
#include <memory>
#include "device.h"
#include "dtype.h"

namespace ormi {
    struct Buffer {
        void* ptr;
        size_t bytes;
        std::vector<int> shape;
        Device device;
        DType dtype;

        Buffer(void* p, size_t b, std::vector<int> s, Device d, DType dt);
        ~Buffer();
        
        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;
    
        float item(size_t index = 0) const;
        size_t numel() const;
    };
    
    std::string to_string(const Buffer& buffer);
    std::string to_string(const std::shared_ptr<Buffer>& buffer);
    std::ostream& operator<<(std::ostream& os, const Buffer& buffer);

    //
    //
    //

    std::shared_ptr<Buffer> empty(const std::vector<int>& shape, Device device, DType dtype);
    std::shared_ptr<Buffer> full(const std::vector<int>& shape, float fill_value, Device device, DType dtype);
    std::shared_ptr<Buffer> scalar(float value, Device device, DType dtype);
    
    std::shared_ptr<Buffer> zeros(const std::vector<int>& shape, Device device, DType dtype);
    std::shared_ptr<Buffer> ones(const std::vector<int>& shape, Device device, DType dtype);

    std::shared_ptr<Buffer> zeros_like(const std::shared_ptr<Buffer>& array, DType dtype);
    std::shared_ptr<Buffer> zeros_like(const std::shared_ptr<Buffer>& array, Device device, DType dtype);

    std::shared_ptr<Buffer> ones_like(const std::shared_ptr<Buffer>& array, DType dtype);
    std::shared_ptr<Buffer> ones_like(const std::shared_ptr<Buffer>& array, Device device, DType dtype);
}