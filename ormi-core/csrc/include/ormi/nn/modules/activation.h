#pragma once
#include <cstdint>
#include <memory>

#include <ormi/core/tensor.h>
#include <ormi/nn/modules/module.h>

namespace ormi::nn {
    class ReLU final : public Module {
    public:
        std::shared_ptr<core::Tensor> forward(const std::shared_ptr<core::Tensor>& input) const override;
    };

    class Sigmoid final : public Module {
    public:
        std::shared_ptr<core::Tensor> forward(const std::shared_ptr<core::Tensor>& input) const override;
    };

    class Tanh final : public Module {
    public:
        std::shared_ptr<core::Tensor> forward(const std::shared_ptr<core::Tensor>& input) const override;
    };

    class Softplus final : public Module {
    public:
        std::shared_ptr<core::Tensor> forward(const std::shared_ptr<core::Tensor>& input) const override;
    };

    class GELU final : public Module {
    public:
        std::shared_ptr<core::Tensor> forward(const std::shared_ptr<core::Tensor>& input) const override;
    };

    class SiLU final : public Module {
    public:
        std::shared_ptr<core::Tensor> forward(const std::shared_ptr<core::Tensor>& input) const override;
    };

    class Softsign final : public Module {
    public:
        std::shared_ptr<core::Tensor> forward(const std::shared_ptr<core::Tensor>& input) const override;
    };

    class Mish final : public Module {
    public:
        std::shared_ptr<core::Tensor> forward(const std::shared_ptr<core::Tensor>& input) const override;
    };

    class Softmax final : public Module {
    public:
        explicit Softmax(std::int64_t axis = -1)
            : axis_(axis) {}

        std::shared_ptr<core::Tensor> forward(const std::shared_ptr<core::Tensor>& input) const override;

    private:
        std::int64_t axis_;
    };

    class LogSoftmax final : public Module {
    public:
        explicit LogSoftmax(std::int64_t axis = -1)
            : axis_(axis) {}

        std::shared_ptr<core::Tensor> forward(const std::shared_ptr<core::Tensor>& input) const override;

    private:
        std::int64_t axis_;
    };

    class LeakyReLU final : public Module {
    public:
        explicit LeakyReLU(double negative_slope = 0.01);
        std::shared_ptr<core::Tensor> forward(const std::shared_ptr<core::Tensor>& input) const override;

    private:
        double negative_slope_;
    };

    class Sin final : public Module {
    public:
        std::shared_ptr<core::Tensor> forward(const std::shared_ptr<core::Tensor>& input) const override;
    };
}
