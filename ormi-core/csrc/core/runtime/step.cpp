#include <algorithm>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
#include <utility>

#include <ormi/core/memory/buffer.h>
#include <ormi/core/op.h>
#include <ormi/backend/api.h>
#include <ormi/core/tensor.h>

#include "core/runtime/step.h"

namespace ormi::core::runtime {
    Step::Step(
        Tensor& output,
        Compute compute,
        Setup setup,
        FusionFn fusion,
        std::optional<fusion::Expr> expression,
        std::vector<Tensor*> auxiliary_outputs
    ) : output_(&output),
        compute_(compute),
        setup_(std::move(setup)),
        fusion_(fusion),
        expression_(std::move(expression)),
        auxiliary_outputs_(std::move(auxiliary_outputs)) {}

    Step::Step(
        Tensor& output,
        PrepareFusionFn prepare,
        PrepareFusionsFn prepare_many,
        FusionFn fusion,
        fusion::Expr expression,
        std::vector<Tensor*> auxiliary_outputs
    ) : output_(&output),
        compute_(nullptr),
        fusion_prepare_(prepare),
        fusion_prepare_many_(prepare_many),
        fusion_(fusion),
        expression_(std::move(expression)),
        auxiliary_outputs_(std::move(auxiliary_outputs)) {}

    Tensor& Step::output() const noexcept {
        return *output_;
    }

    std::size_t Step::output_count() const noexcept {
        return 1 + auxiliary_outputs_.size();
    }

    Tensor& Step::output(std::size_t index) const noexcept {
        return index == 0 ? *output_ : *auxiliary_outputs_[index - 1];
    }

    std::size_t Step::input_count() const noexcept {
        return expression_
            ? expression_->inputs.size()
            : output_->parents_.size();
    }

    Tensor& Step::input(std::size_t index) const noexcept {
        return expression_
            ? *expression_->inputs[index]
            : *output_->parents_[index];
    }

    bool Step::allocates() const noexcept {
        return compute_ || setup_ || fusion_;
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

    void Step::prepared() noexcept {
        fusion_prepare_ = nullptr;
        fusion_prepare_many_ = nullptr;
    }

    void Step::prepare() {
        if (!fusion_prepare_) {
            return;
        }

        auto setup = fusion_prepare_(
            *output_,
            auxiliary_outputs_,
            *expression_
        );
        setup_ = std::move(setup);
        prepared();
    }

    void prepare(std::span<Step> steps) {
        struct Batch {
            PrepareFusionsFn prepare;
            std::vector<Step*> steps;
            std::vector<FusionJob> jobs;
        };

        std::vector<Batch> batches;
        for (auto& step : steps) {
            if (!step.fusion_prepare_many_) {
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
                step->prepared();
            }
        }
    }

    void Step::run() {
        prepare();

        if (!allocates()) {
            output_->buffer_ = output_->parents_[0]->buffer_;
            return;
        }

        if (!auxiliary_outputs_.empty()) {
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
                if (setup_) {
                    setup_.run(*output_);
                } else {
                    fusion_(*output_, auxiliary_outputs_, *expression_);
                }
            } catch (...) {
                for (auto* output : allocated) {
                    output->buffer_.reset();
                }
                throw;
            }
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
            if (setup_) {
                setup_.run(*output_);
            } else if (fusion_) {
                fusion_(*output_, {}, *expression_);
            } else {
                compute_(*output_);
            }
        } catch (...) {
            if (allocated) {
                output_->buffer_.reset();
            }
            throw;
        }
    }

}
