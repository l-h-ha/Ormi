#pragma once

#include <ormi/backend/fusion.h>

namespace ormi::backends::cuda::fusion {
    bool physical_order(
        const core::runtime::fusion::Expr& expression
    );
}
