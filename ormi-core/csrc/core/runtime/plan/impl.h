#pragma once
#include <cstddef>
#include <memory>
#include <vector>

#include <ormi/backend/api.h>
#include <ormi/core/device.h>
#include <ormi/core/plan.h>

#include "core/runtime/step.h"

namespace ormi::core {
    class Plan::Impl final {
    public:
        Impl(
            std::vector<std::shared_ptr<Tensor>> outputs,
            std::vector<Update> updates
        );

        void run();
        Replay record(std::shared_ptr<Impl> owner);
        std::size_t steps() const noexcept;
        std::size_t storage_buffers() const noexcept;
        std::size_t storage_bytes() const noexcept;

    private:
        struct Copy {
            Tensor* destination;
            Tensor* value;
            runtime::CopyAsyncFn execute;
        };

        void append_update(const Update& update);
        void count_storage();
        Device recording_device() const;

        // Roots own the graph. Steps and copies borrow its tensors.
        std::vector<std::shared_ptr<Tensor>> outputs_;
        std::vector<Update> updates_;
        std::vector<runtime::Step> steps_;
        std::vector<Copy> copies_;
        std::size_t storage_buffers_ = 0;
        std::size_t storage_bytes_ = 0;
        bool ran_ = false;
    };
}
