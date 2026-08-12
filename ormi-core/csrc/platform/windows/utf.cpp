#include <string>
#include <string_view>

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <ormi/core/error.h>

#include "platform/windows/utf.h"

namespace ormi::platform::windows {
    std::wstring widen(std::string_view text) {
        if (text.empty()) {
            return {};
        }

        const auto size = MultiByteToWideChar(
            CP_UTF8,
            MB_ERR_INVALID_CHARS,
            text.data(),
            static_cast<int>(text.size()),
            nullptr,
            0
        );
        if (size <= 0) {
            ORMI_THROW(jitError, "Cannot convert UTF-8 text to UTF-16.");
        }

        std::wstring result(static_cast<std::size_t>(size), L'\0');
        MultiByteToWideChar(
            CP_UTF8,
            MB_ERR_INVALID_CHARS,
            text.data(),
            static_cast<int>(text.size()),
            result.data(),
            size
        );
        return result;
    }
}
