#include <cstddef>

#include <ormi/core/plan.h>

#include "core/runtime/plan/impl.h"

namespace ormi::core {
    Plan::~Plan() = default;
    Plan::Plan(Plan&&) noexcept = default;
    Plan& Plan::operator=(Plan&&) noexcept = default;

    std::size_t Plan::steps() const noexcept {
        return impl_->steps();
    }

    std::size_t Plan::storage_buffers() const noexcept {
        return impl_->storage_buffers();
    }

    std::size_t Plan::storage_bytes() const noexcept {
        return impl_->storage_bytes();
    }

    std::size_t Plan::Impl::steps() const noexcept {
        return steps_.size();
    }

    std::size_t Plan::Impl::storage_buffers() const noexcept {
        return storage_buffers_;
    }

    std::size_t Plan::Impl::storage_bytes() const noexcept {
        return storage_bytes_;
    }
}
