#include <type_traits>

#include <ormi/core/setup.h>

static_assert(std::is_default_constructible_v<ormi::core::Setup>);
static_assert(std::is_nothrow_move_constructible_v<ormi::core::Setup>);
static_assert(std::is_nothrow_move_assignable_v<ormi::core::Setup>);
static_assert(!std::is_copy_constructible_v<ormi::core::Setup>);
static_assert(!std::is_copy_assignable_v<ormi::core::Setup>);
