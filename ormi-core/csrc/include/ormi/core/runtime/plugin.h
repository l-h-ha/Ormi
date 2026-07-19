#pragma once

#include <string_view>

#include <ormi/core/runtime/backend.h>

namespace ormi::core::runtime::plugin {
    using Init = const Backend* (*)() noexcept;

    inline constexpr std::string_view entry = "ormi_backend_init";
}
