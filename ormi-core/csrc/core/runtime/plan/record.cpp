#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <ormi/core/error.h>
#include <ormi/core/tensor.h>

#include "core/runtime/device.h"
#include "core/runtime/plan/impl.h"

namespace ormi::core {
    class Plan::Replay::Impl final {
    public:
        Impl(
            std::shared_ptr<Plan::Impl> plan,
            void* recording,
            runtime::ReplayFn replay,
            runtime::ReleaseFn release
        ) : plan_(std::move(plan)),
            recording_(recording),
            replay_(replay),
            release_(release) {}

        ~Impl() {
            release_(recording_);
        }

        void run() {
            replay_(recording_);
        }

    private:
        // The recording borrows tensor storage owned by the plan.
        std::shared_ptr<Plan::Impl> plan_;
        void* recording_;
        runtime::ReplayFn replay_;
        runtime::ReleaseFn release_;
    };

    Plan::Replay Plan::record() {
        return impl_->record(impl_);
    }

    Plan::Replay::Replay(std::unique_ptr<Impl> impl)
        : impl_(std::move(impl)) {}

    Plan::Replay::~Replay() = default;
    Plan::Replay::Replay(Replay&&) noexcept = default;
    Plan::Replay& Plan::Replay::operator=(Replay&&) noexcept = default;

    void Plan::Replay::run() {
        impl_->run();
    }

    Plan::Replay Plan::Impl::record(std::shared_ptr<Impl> owner) {
        const auto device = recording_device();
        const auto& backend = runtime::backend(device);
        if (!backend.recording.record) {
            ORMI_THROW(
                OperationError,
                "Backend '" + std::string(device.name()) +
                    "' does not support recording."
            );
        }
        if (!ran_) {
            ORMI_THROW(
                OperationError,
                "A plan must run successfully before recording."
            );
        }

        void* recording = backend.recording.record(
            device,
            [](void* context) {
                static_cast<Impl*>(context)->run();
            },
            this
        );
        if (!recording) {
            ORMI_THROW(
                OperationError,
                "Backend '" + std::string(device.name()) +
                    "' returned an empty recording."
            );
        }

        try {
            return Replay(std::make_unique<Replay::Impl>(
                std::move(owner),
                recording,
                backend.recording.replay,
                backend.recording.release
            ));
        } catch (...) {
            backend.recording.release(recording);
            throw;
        }
    }

    Device Plan::Impl::recording_device() const {
        std::optional<Device> device;

        const auto include = [&device](Device candidate) {
            if (!device) {
                device = candidate;
                return;
            }
            if (*device != candidate) {
                ORMI_THROW(
                    OperationError,
                    "Recording requires one device."
                );
            }
        };

        for (const auto& step : steps_) {
            include(step.output().device());
        }
        for (const auto& copy : copies_) {
            include(copy.destination->device_);
        }

        if (!device) {
            ORMI_THROW(
                OperationError,
                "Cannot record a plan with no executable steps."
            );
        }
        return *device;
    }
}
