#pragma once
#include <string_view>

#include <ormi/core/runtime/backend_id.h>

namespace ormi::core::runtime {
    struct Backend;
}

namespace ormi::core {
    class Device {
    public:
        Device();
        Device(runtime::BackendId backend_id, int index = 0);

        static Device cpu();
        static Device cuda(int index = 0);
        static Device from(std::string_view device_string);

        runtime::BackendId backend_id() const;
        const runtime::Backend* backend() const;
        int index() const;

        bool is_cpu() const;
        bool is_cuda() const;

    private:
        runtime::BackendId backend_id_;
        const runtime::Backend* backend_;
        int index_;
    };
}
