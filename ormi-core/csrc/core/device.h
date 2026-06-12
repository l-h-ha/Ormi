#pragma once
#include <string>
#include <stdexcept>

namespace ormi {
    enum class DeviceEnum { CPU, CUDA };
    
    struct Device {
        DeviceEnum type;
        int id;
    
        Device(DeviceEnum t = DeviceEnum::CUDA, int i = 0) : type(t), id(i) {}
    
        std::string to_string() const {
            std::string t;
            switch (type) {
                case DeviceEnum::CPU:  t = "cpu"; break;
                case DeviceEnum::CUDA: t = "cuda"; break;
                default: throw std::runtime_error("ormi::Device - Unknown DeviceEnum");
            }
            return t + ":" + std::to_string(id);
        }
        
        bool operator==(const Device& other) const {
            return type == other.type && id == other.id;
        }

        bool operator!=(const Device& other) const {
            return !(*this == other);
        }
    };
}