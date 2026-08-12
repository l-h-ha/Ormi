#pragma once

#include <string_view>

#include <ormi/backend/api.h>

namespace ormi::core::runtime {
    Backend normalize(Backend backend, std::string_view name);
}
