#pragma once

#include <algorithm>
#include <cctype>
#include <ostream>
#include <string>

#include <magic_enum.hpp>

#include "core/error.h"

namespace ormi {
    enum class DeviceEnum {
        CPU,
        CUDA
    };

    struct Device {
        DeviceEnum type = DeviceEnum::CPU;
        int id = 0;

        Device() = default;

        explicit Device(DeviceEnum t, int i = 0)
            : type(t), id(i) {
            if (id < 0) {
                ORMI_THROW(DeviceError, "Device id cannot be negative.");
            }

            if (type == DeviceEnum::CPU && id != 0) {
                ORMI_THROW(DeviceError, "CPU device id must be 0.");
            }
        }

        static Device cpu() {
            return Device(DeviceEnum::CPU, 0);
        }

        static Device cuda(int id = 0) {
            return Device(DeviceEnum::CUDA, id);
        }

        bool is_device_enum(const DeviceEnum dev_type) {
            return this->type == dev_type;
        }

        bool operator==(const Device& other) const {
            return type == other.type && id == other.id;
        }

        bool operator!=(const Device& other) const {
            return !(*this == other);
        }
    };

    inline std::string to_string(DeviceEnum dev_type) {
        auto name = magic_enum::enum_name(dev_type);
        if (name.empty()) {
            ORMI_THROW(DeviceError, "Unrecognized DeviceEnum");
        }

        std::string out(name);
        std::transform(
            out.begin(),
            out.end(),
            out.begin(),
            [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            }
        );

        return out;
    }

    inline std::string to_string(const Device& dev) {
        return to_string(dev.type) + ":" + std::to_string(dev.id);
    }

    inline std::ostream& operator<<(std::ostream& os, const Device& dev) {
        return os << to_string(dev);
    }

    inline std::ostream& operator<<(std::ostream& os, DeviceEnum dev_type) {
        return os << to_string(dev_type);
    }
}