#include <ormi/core/op.h>

static_assert(sizeof(ormi::core::Setup) > 0);
static_assert(sizeof(ormi::core::OpRef) == sizeof(const ormi::core::Op*));
