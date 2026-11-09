#include <limits>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ormi/core/error.h>
#include <ormi/core/tensor.h>

#include "core/runtime/device.h"
#include "core/runtime/fusion/group.h"
#include "core/runtime/plan/impl.h"
#include "core/runtime/schedule.h"

namespace ormi::core {
    Plan::Plan(
        std::vector<std::shared_ptr<Tensor>> outputs,
        std::vector<Update> updates
    ) : impl_(std::make_shared<Impl>(
        std::move(outputs),
        std::move(updates)
    )) {}

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
            append_update(update);
            root_tensors.push_back(update.value.get());
        }

        const auto candidates = runtime::fusion::find_candidates(root_tensors);
        for (auto* root : root_tensors) {
            runtime::schedule::append(*root, planned, steps_, candidates);
        }

        runtime::prepare(steps_);
        count_storage();
    }

    void Plan::Impl::count_storage() {
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

        for (const auto& step : steps_) {
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

    void Plan::Impl::append_update(const Update& update) {
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
        if (update.destination->device_ != update.value->device_) {
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
}
