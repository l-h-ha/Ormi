#pragma once

#include <cstddef>

namespace ormi::backends::cpu::avx2 {
    void sum_rows(
        float* sums,
        const float* input,
        std::size_t rows,
        std::size_t columns
    );
}
