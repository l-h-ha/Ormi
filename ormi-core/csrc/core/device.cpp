#include <string>
#include <string_view>
#include <stdexcept>

#include <ormi/core/device.h>
#include <ormi/core/error.h>
#include <ormi/core/runtime/backend_id.h>
#include <ormi/core/runtime/backends.h>

namespace ormi::core {
    namespace {
        int parse_device_index(std::string_view index_text, std::string_view device_string) {
            if (index_text.empty()) {
                ORMI_THROW(DeviceError, "Invalid device string: " + std::string(device_string));
            }

            try {
                size_t consumed = 0;
                const int index = std::stoi(std::string(index_text), &consumed);

                if (consumed != index_text.size()) {
                    ORMI_THROW(DeviceError, "Invalid device string: " + std::string(device_string));
                }

                return index;
            } catch (const std::invalid_argument&) {
                ORMI_THROW(DeviceError, "Invalid device string: " + std::string(device_string));
            } catch (const std::out_of_range&) {
                ORMI_THROW(DeviceError, "Invalid device string: " + std::string(device_string));
            }
        }
    }

    Device::Device() : Device(runtime::backend_id_cpu, 0) {}

    Device::Device(runtime::BackendId backend_id, int index)
        : backend_id_(backend_id), backend_(nullptr), index_(index)
    {
        if (index < 0) {
            ORMI_THROW(DeviceError, "Device index cannot be negative.");
        }

        if (backend_id == runtime::backend_invalid) {
            ORMI_THROW(DeviceError, "Invalid backend id.");
        }

        if (backend_id == runtime::backend_id_cpu && index != 0) {
            ORMI_THROW(DeviceError, "CPU device index must be 0.");
        }

        backend_ = &runtime::Backends::instance().require(backend_id);
    }

    Device Device::cpu() {
        return Device(runtime::backend_id_cpu, 0);
    }

    Device Device::cuda(int index) {
        return Device(runtime::backend_id_cuda, index);
    }

    Device Device::from(std::string_view device_string) {
        std::string_view backend_name = device_string;
        int index = 0;

        const size_t separator = device_string.find(':');

        if (separator != std::string_view::npos) {
            backend_name = device_string.substr(0, separator);
            index = parse_device_index(device_string.substr(separator + 1), device_string);
        }

        if (backend_name.empty()) {
            ORMI_THROW(DeviceError, "Invalid device string: " + std::string(device_string));
        }

        const runtime::BackendId id = runtime::Backends::instance().find(backend_name);

        if (id == runtime::backend_invalid) {
            ORMI_THROW(DeviceError, "Invalid device string: " + std::string(device_string));
        }

        return Device(id, index);
    }

    runtime::BackendId Device::backend_id() const {
        return backend_id_;
    }

    const runtime::Backend* Device::backend() const {
        return backend_;
    }

    int Device::index() const {
        return index_;
    }

    bool Device::is_cpu() const {
        return backend_id_ == runtime::backend_id_cpu;
    }

    bool Device::is_cuda() const {
        return backend_id_ == runtime::backend_id_cuda;
    }
}
