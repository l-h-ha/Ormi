#pragma once

#include <string>

namespace ormi::backends::cuda::fusion::nvrtc {
    int target(int major, int minor);
    std::string context(int major, int minor);
    std::string compile(const std::string& source, int major, int minor);
}
