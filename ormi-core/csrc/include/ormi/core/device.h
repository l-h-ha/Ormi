#pragma once
#include <string_view>

#include <ormi/backend/id.h>

namespace ormi::core {
    class Device;
}

namespace ormi::core::runtime {
    struct Backend;
    const Backend& backend(const Device& device);
}

namespace ormi::core {
    class Device {
    public:
        Device();
        Device(runtime::BackendId backend_id, int index = 0);

        static Device cpu();
        static Device from(std::string_view device_string);
        static Device from(std::string_view backend_name, int index);

        runtime::BackendId backend_id() const;
        std::string_view name() const;
        int index() const;

        bool indexed() const;
        bool is_host() const;

    private:
        friend const runtime::Backend& runtime::backend(const Device& device);

        runtime::BackendId backend_id_;
        const runtime::Backend* backend_;
        int index_;
    };
}
