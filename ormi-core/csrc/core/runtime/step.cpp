#include <algorithm>
#include <iterator>
#include <memory>
#include <span>
#include <utility>

#include <ormi/core/memory/buffer.h>
#include <ormi/core/op.h>
#include <ormi/backend/api.h>
#include <ormi/core/tensor.h>

#include "core/runtime/step.h"

namespace ormi::core::runtime {
    Step::Step(Tensor& output) noexcept
        : output_(&output), kind_(Kind::alias) {}

    Step::Step(Tensor& output, Compute compute, Setup setup) noexcept
        : output_(&output),
          kind_(Kind::compute),
          compute_(compute),
          setup_(std::move(setup)) {}

    Step::Step(
        Tensor& output,
        FusionFn execute,
        fusion::Expr expression,
        std::vector<Tensor*> auxiliary_outputs
    ) noexcept : output_(&output),
        kind_(Kind::fusion),
        fusion_(execute),
        expression_(std::move(expression)),
        auxiliary_outputs_(std::move(auxiliary_outputs)) {}

    Step Step::alias(Tensor& output) {
        return Step(output);
    }

    Step Step::compute(
        Tensor& output,
        Compute compute,
        Setup setup
    ) {
        return Step(output, compute, std::move(setup));
    }

    Step Step::fusion(
        Tensor& output,
        FusionFn execute,
        fusion::Expr expression,
        std::vector<Tensor*> auxiliary_outputs
    ) {
        return Step(
            output,
            execute,
            std::move(expression),
            std::move(auxiliary_outputs)
        );
    }

    Step Step::deferred_fusion(
        Tensor& output,
        PrepareFusionFn prepare,
        PrepareFusionsFn prepare_many,
        FusionFn execute,
        fusion::Expr expression,
        std::vector<Tensor*> auxiliary_outputs
    ) {
        auto step = fusion(
            output,
            execute,
            std::move(expression),
            std::move(auxiliary_outputs)
        );
        step.preparation_pending_ = true;
        step.fusion_prepare_ = prepare;
        step.fusion_prepare_many_ = prepare_many;
        return step;
    }

    Tensor& Step::output() const noexcept {
        return *output_;
    }

    std::size_t Step::output_count() const noexcept {
        return kind_ == Kind::fusion
            ? 1 + auxiliary_outputs_.size()
            : 1;
    }

    Tensor& Step::output(std::size_t index) const noexcept {
        return index == 0 ? *output_ : *auxiliary_outputs_[index - 1];
    }

    std::size_t Step::input_count() const noexcept {
        return kind_ == Kind::fusion
            ? expression_->inputs.size()
            : output_->parents_.size();
    }

    Tensor& Step::input(std::size_t index) const noexcept {
        return kind_ == Kind::fusion
            ? *expression_->inputs[index]
            : *output_->parents_[index];
    }

    bool Step::allocates() const noexcept {
        return kind_ != Kind::alias;
    }

    std::size_t Step::setup_buffers() const noexcept {
        return setup_.buffers();
    }

    std::size_t Step::setup_bytes() const noexcept {
        return setup_.bytes();
    }

    FusionJob Step::job() noexcept {
        return FusionJob{
            output_,
            auxiliary_outputs_.data(),
            auxiliary_outputs_.size(),
            &*expression_,
            &setup_
        };
    }

    void Step::prepare() {
        if (!preparation_pending_) {
            return;
        }

        if (fusion_prepare_) {
            setup_ = fusion_prepare_(
                *output_,
                auxiliary_outputs_,
                *expression_
            );
        } else {
            // A failed batch may have installed a setup before direct fallback.
            setup_ = {};
        }
        preparation_pending_ = false;
    }

    void prepare(std::span<Step> steps) {
        struct Batch {
            PrepareFusionsFn prepare;
            std::vector<Step*> steps;
            std::vector<FusionJob> jobs;
        };

        std::vector<Batch> batches;
        for (auto& step : steps) {
            if (
                !step.preparation_pending_ ||
                !step.fusion_prepare_many_
            ) {
                step.prepare();
                continue;
            }

            auto batch = std::find_if(
                batches.begin(),
                batches.end(),
                [&step](const Batch& candidate) {
                    return candidate.prepare == step.fusion_prepare_many_;
                }
            );
            if (batch == batches.end()) {
                batches.push_back(Batch{step.fusion_prepare_many_, {}, {}});
                batch = std::prev(batches.end());
            }
            batch->steps.push_back(&step);
            batch->jobs.push_back(step.job());
        }

        for (auto& batch : batches) {
            batch.prepare(batch.jobs);
            for (auto* step : batch.steps) {
                step->preparation_pending_ = false;
            }
        }
    }

    void Step::execute() {
        if (setup_) {
            setup_.run(*output_);
        } else if (kind_ == Kind::fusion) {
            fusion_(*output_, auxiliary_outputs_, *expression_);
        } else {
            compute_(*output_);
        }
    }

    void Step::allocate_group_and_run() {
        std::vector<Tensor*> allocated;
        allocated.reserve(output_count());
        const auto allocate = [&allocated](Tensor& output) {
            if (output.buffer_) {
                return;
            }
            output.buffer_ = std::make_shared<memory::Buffer>(
                output.device_,
                output.bytes()
            );
            allocated.push_back(&output);
        };

        try {
            allocate(*output_);
            for (auto* output : auxiliary_outputs_) {
                allocate(*output);
            }
            execute();
        } catch (...) {
            for (auto* output : allocated) {
                output->buffer_.reset();
            }
            throw;
        }
    }

    void Step::run() {
        prepare();

        if (kind_ == Kind::alias) {
            output_->buffer_ = output_->parents_[0]->buffer_;
            return;
        }

        if (kind_ == Kind::fusion && !auxiliary_outputs_.empty()) {
            if (
                output_->buffer_ &&
                std::all_of(
                    auxiliary_outputs_.begin(),
                    auxiliary_outputs_.end(),
                    [](const Tensor* output) { return output->buffer_ != nullptr; }
                )
            ) {
                execute();
                return;
            }

            allocate_group_and_run();
            return;
        }

        const bool allocated = !output_->buffer_;
        if (allocated) {
            output_->buffer_ = std::make_shared<memory::Buffer>(
                output_->device_,
                output_->bytes()
            );
        }

        try {
            execute();
        } catch (...) {
            if (allocated) {
                output_->buffer_.reset();
            }
            throw;
        }
    }

}
