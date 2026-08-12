#pragma once
#include <string>
#include <string_view>

namespace ormi::platform::windows {
    std::wstring widen(std::string_view text);
}
