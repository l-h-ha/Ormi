#pragma once

#include <ormi/core/dtype.h>

namespace ormi::core {
    class Autocast {
    public:
        explicit Autocast(DType dtype = DType::float16);
        ~Autocast();

        Autocast(const Autocast&) = delete;
        Autocast& operator=(const Autocast&) = delete;
        Autocast(Autocast&&) = delete;
        Autocast& operator=(Autocast&&) = delete;

    private:
        bool previous_enabled_;
        DType previous_dtype_;
    };
}
