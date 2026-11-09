#pragma once
#include <cstdint>
#include <memory>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/tensor.h>
#include <ormi/nn/modules/module.h>

namespace ormi::nn {
    // Normalize every axis except the channel axis, whose size must be num_features.
    // Default layout: [B, ..., C]. Set axis=1 for [B, C, ...]. The output shape is unchanged.
    // Training uses population variance. running_var stores an unbiased variance estimate.
    // Float16 statistics and running buffers use Float32. Other floating dtypes keep their precision.
    class BatchNorm final : public Module {
    public:
        explicit BatchNorm(
            std::int64_t num_features,
            double eps = 1e-5,
            double momentum = 0.1,
            bool affine = true,
            bool track_running_stats = true,
            std::int64_t axis = -1,
            core::DType dtype = core::DType::float32,
            core::Device device = core::Device::cpu()
        );
        // Nonempty batch-statistics mode requires at least two values per channel.
        // Running updates are lazy: running = (1-momentum)*running + momentum*batch.
        // With track_running_stats=false, evaluation also uses batch statistics.
        std::shared_ptr<core::Tensor> forward(const std::shared_ptr<core::Tensor>& input) const override;

        std::int64_t num_features() const noexcept;
        std::int64_t axis() const noexcept;
        double eps() const noexcept;
        double momentum() const noexcept;
        std::shared_ptr<core::Tensor> weight() const noexcept;
        std::shared_ptr<core::Tensor> bias() const noexcept;
        std::shared_ptr<core::Tensor> running_mean() const noexcept;
        std::shared_ptr<core::Tensor> running_var() const noexcept;

    private:
        std::int64_t num_features_;
        double eps_;
        double momentum_;
        std::int64_t axis_;
        std::shared_ptr<core::Tensor> weight_;
        std::shared_ptr<core::Tensor> bias_;
        std::shared_ptr<core::Tensor> running_mean_;
        std::shared_ptr<core::Tensor> running_var_;
        // Link unexecuted forwards in call order. A strong member reference would retain their graphs.
        mutable std::weak_ptr<core::Tensor> previous_;
    };
}
