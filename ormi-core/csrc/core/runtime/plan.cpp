#include <memory>
#include <limits>
#include <span>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ormi/core/error.h>
#include <ormi/core/plan.h>
#include <ormi/backend/api.h>
#include <ormi/core/tensor.h>

#include "core/runtime/device.h"
#include "core/runtime/fusion/group.h"
#include "core/runtime/schedule.h"
#include "core/runtime/step.h"

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
        std::shared_ptr<Plan::Impl> plan_;
        void* recording_;
        runtime::ReplayFn replay_;
        runtime::ReleaseFn release_;
    };

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

        void append(Update update);
        Device recording_device() const;

        std::vector<std::shared_ptr<Tensor>> outputs_;
        std::vector<Update> updates_;
        std::vector<runtime::Step> steps_;
        std::vector<Copy> copies_;
        std::size_t storage_buffers_ = 0;
        std::size_t storage_bytes_ = 0;
        bool ran_ = false;
    };

    Plan::Plan(
        std::vector<std::shared_ptr<Tensor>> outputs,
        std::vector<Update> updates
    ) : impl_(std::make_shared<Impl>(
        std::move(outputs),
        std::move(updates)
    )) {}

    Plan::~Plan() = default;
    Plan::Plan(Plan&&) noexcept = default;
    Plan& Plan::operator=(Plan&&) noexcept = default;

    void Plan::run() {
        impl_->run();
    }

    Plan::Replay Plan::record() {
        return impl_->record(impl_);
    }

    std::size_t Plan::steps() const noexcept {
        return impl_->steps();
    }

    std::size_t Plan::storage_buffers() const noexcept {
        return impl_->storage_buffers();
    }

    std::size_t Plan::storage_bytes() const noexcept {
        return impl_->storage_bytes();
    }

    Plan::Replay::Replay(std::unique_ptr<Impl> impl)
        : impl_(std::move(impl)) {}

    Plan::Replay::~Replay() = default;
    Plan::Replay::Replay(Replay&&) noexcept = default;
    Plan::Replay& Plan::Replay::operator=(Replay&&) noexcept = default;

    void Plan::Replay::run() {
        impl_->run();
    }

    Plan::Impl::Impl(
        std::vector<std::shared_ptr<Tensor>> outputs,
        std::vector<Update> updates
    ) : outputs_(std::move(outputs)), updates_(std::move(updates)) {
        if (outputs_.empty() && updates_.empty()) {
            ORMI_THROW(
                OperationError,
                "A plan requires an output or update."
            );
        }

        const auto roots = outputs_.size() + updates_.size();
        steps_.reserve(roots);
        std::unordered_set<const Tensor*> planned;
        std::vector<Tensor*> root_tensors;
        root_tensors.reserve(roots);

        for (const auto& output : outputs_) {
            if (!output) {
                ORMI_THROW(
                    OperationError,
                    "A plan cannot contain a null output."
                );
            }
            if (output->buffer_) {
                ORMI_THROW(
                    OperationError,
                    "A plan output must be lazy."
                );
            }
            root_tensors.push_back(output.get());
        }

        copies_.reserve(updates_.size());
        std::unordered_set<const Tensor*> destinations;
        destinations.reserve(updates_.size());
        for (const auto& update : updates_) {
            if (
                update.destination &&
                !destinations.insert(update.destination.get()).second
            ) {
                ORMI_THROW(
                    OperationError,
                    "A plan cannot update one destination twice."
                );
            }
            append(update);
            root_tensors.push_back(update.value.get());
        }

        const auto grouped = runtime::fusion::groups(root_tensors);
        for (const auto& output : outputs_) {
            runtime::schedule::append(
                *output,
                planned,
                steps_,
                grouped
            );
        }
        for (const auto& update : updates_) {
            runtime::schedule::append(
                *update.value,
                planned,
                steps_,
                grouped
            );
        }

        runtime::prepare(steps_);

        for (const auto& step : steps_) {
            const auto add_storage = [this](
                std::size_t buffers,
                std::size_t bytes
            ) {
                if (
                    storage_buffers_ >
                    std::numeric_limits<std::size_t>::max() - buffers ||
                    storage_bytes_ >
                    std::numeric_limits<std::size_t>::max() - bytes
                ) {
                    ORMI_THROW(
                        MemoryError,
                        "Plan storage size overflow."
                    );
                }

                storage_buffers_ += buffers;
                storage_bytes_ += bytes;
            };

            if (step.allocates()) {
                for (
                    std::size_t output = 0;
                    output < step.output_count();
                    ++output
                ) {
                    const auto bytes = step.output(output).bytes();
                    if (bytes != 0) {
                        add_storage(1, bytes);
                    }
                }
            }

            add_storage(step.setup_buffers(), step.setup_bytes());
        }
    }

    void Plan::Impl::append(Update update) {
        if (!update.destination || !update.value) {
            ORMI_THROW(
                OperationError,
                "A plan update requires a destination and value."
            );
        }
        if (!update.destination->buffer_) {
            ORMI_THROW(
                BufferError,
                "A plan update destination must be realized."
            );
        }
        if (
            update.destination->device_.backend_id() !=
                update.value->device_.backend_id() ||
            update.destination->device_.index() !=
                update.value->device_.index()
        ) {
            ORMI_THROW(
                DeviceError,
                "A plan update requires one device."
            );
        }
        if (update.destination->dtype_ != update.value->dtype_) {
            ORMI_THROW(
                DTypeError,
                "A plan update requires matching dtypes."
            );
        }
        if (update.destination->shape_ != update.value->shape_) {
            ORMI_THROW(
                ShapeError,
                "A plan update requires matching shapes."
            );
        }
        if (
            !update.destination->is_contiguous() ||
            !update.value->is_contiguous()
        ) {
            ORMI_THROW(
                OperationError,
                "A plan update requires contiguous tensors."
            );
        }

        copies_.push_back(Copy{
            update.destination.get(),
            update.value.get(),
            runtime::backend(update.destination->device_).memory.copy_async
        });
    }

    void Plan::Impl::run() {
        for (auto& step : steps_) {
            step.run();
        }

        for (const auto& copy : copies_) {
            const auto bytes = copy.destination->bytes();
            if (bytes == 0) {
                continue;
            }

            auto* destination = copy.destination->data();
            const auto* value = copy.value->data();
            if (destination == value) {
                continue;
            }
            if (copy.destination->buffer_ == copy.value->buffer_) {
                ORMI_THROW(
                    OperationError,
                    "A plan update cannot copy between overlapping aliases."
                );
            }

            copy.execute(
                copy.destination->device_,
                destination,
                value,
                bytes
            );
        }

        ran_ = true;
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
            if (
                device->backend_id() != candidate.backend_id() ||
                device->index() != candidate.index()
            ) {
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
