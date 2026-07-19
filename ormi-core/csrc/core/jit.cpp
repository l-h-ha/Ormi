#include <ormi/core/jit.h>

#include "backends/cpu/jit/cache.h"

namespace ormi::core::jit {
    namespace {
        CacheInfo public_info(const backends::cpu::jit::cache::Info& value) {
            return CacheInfo{
                value.path,
                value.entries,
                value.bytes,
                value.limit
            };
        }
    }

    CacheInfo info() {
        return public_info(backends::cpu::jit::cache::info());
    }

    CacheInfo clear() {
        return public_info(backends::cpu::jit::cache::clear());
    }
}
