#include <string>
#include <string_view>
#include <stdexcept>

#include <ormi/core/device.h>
#include <ormi/core/error.h>
#include <ormi/backend/id.h>

#include "core/runtime/backends.h"
#include "core/runtime/catalog.h"

namespace ormi::core {
    namespace {
        constexpr auto host_backend = runtime::catalog::host();

        static_assert(host_backend != runtime::backend_invalid);

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

    Device::Device() : Device(host_backend, 0) {}

    Device::Device(runtime::BackendId backend_id, int index)
        : backend_id_(backend_id), backend_(nullptr), index_(index)
    {
        if (index < 0) {
            ORMI_THROW(DeviceError, "Device index cannot be negative.");
        }

        if (
            backend_id == runtime::backend_invalid ||
            backend_id >= runtime::catalog::entries.size()
        ) {
            ORMI_THROW(DeviceError, "Invalid backend id.");
        }

        const auto& entry = runtime::catalog::entries[backend_id];
        if (!entry.indexed && index != 0) {
            ORMI_THROW(
                DeviceError,
                "Backend '" + std::string(entry.name) +
                    "' device index must be 0."
            );
        }

        backend_ = &runtime::Backends::instance().require(backend_id);
    }

    Device Device::cpu() {
        return Device(host_backend, 0);
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

        return from(backend_name, index);
    }

    Device Device::from(std::string_view backend_name, int index) {
        if (backend_name.empty() || backend_name.find(':') != std::string_view::npos) {
            ORMI_THROW(DeviceError, "Invalid backend name: " + std::string(backend_name));
        }

        const runtime::BackendId id = runtime::Backends::instance().find(backend_name);

        if (id == runtime::backend_invalid) {
            ORMI_THROW(DeviceError, "Unknown backend: " + std::string(backend_name));
        }

        return Device(id, index);
    }

    runtime::BackendId Device::backend_id() const {
        return backend_id_;
    }

    std::string_view Device::name() const {
        return runtime::catalog::entries[backend_id_].name;
    }

    int Device::index() const {
        return index_;
    }

    bool Device::indexed() const {
        return runtime::catalog::entries[backend_id_].indexed;
    }

    bool Device::is_host() const {
        return backend_id_ == host_backend;
    }
}
