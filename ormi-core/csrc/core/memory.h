#pragma once
#include <string>
#include <iostream>
#include <vector>
#include <memory>
#include "../core/device.h"

namespace ormi {
    struct Buffer {
        void* ptr;
        size_t bytes;
        std::vector<int> shape;
        Device device;

        Buffer(void* p, size_t b, std::vector<int> s, Device d);
        ~Buffer();
        
        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;
    };

    std::string to_string(const Buffer& buffer);
    std::string to_string(const std::shared_ptr<Buffer>& buffer);
    std::ostream& operator<<(std::ostream& os, const Buffer& buffer);

    std::shared_ptr<Buffer> empty(std::vector<int> shape, DeviceEnum dev_type);
    std::shared_ptr<Buffer> full(std::vector<int> shape, float fill_value, DeviceEnum dev_type);
    std::shared_ptr<Buffer> scalar(float val, DeviceEnum dev_type);

    std::shared_ptr<Buffer> zeros(std::vector<int> shape, DeviceEnum dev_type);
    std::shared_ptr<Buffer> ones(std::vector<int> shape, DeviceEnum dev_type);

    std::shared_ptr<Buffer> zeros_like(std::shared_ptr<Buffer> array);
    std::shared_ptr<Buffer> zeros_like(std::shared_ptr<Buffer> array, DeviceEnum dev_type);

    std::shared_ptr<Buffer> ones_like(std::shared_ptr<Buffer> array);
    std::shared_ptr<Buffer> ones_like(std::shared_ptr<Buffer> array, DeviceEnum dev_type);
}