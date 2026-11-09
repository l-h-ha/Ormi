#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/memory/ops.h>
#include <ormi/core/ops/linalg.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/nn/modules/linear.h>

namespace ormi::nn {
    namespace {
        template <typename T>
        void fill_uniform(core::Tensor& tensor, std::mt19937_64& generator, double bound) {
            std::uniform_real_distribution<double> distribution(-bound, bound);
            std::vector<T> values(tensor.numel());

            for (auto& value : values) {
                value = static_cast<T>(distribution(generator));
            }

            core::memory::copy(
                tensor.data(),
                tensor.device(),
                values.data(),
                core::Device::cpu(),
                tensor.bytes()
            );
        }

        std::shared_ptr<core::Tensor> create_parameter(
            core::Shape shape,
            core::DType dtype,
            core::Device device,
            std::mt19937_64& generator,
            double bound
        ) {
            auto tensor = std::make_shared<core::Tensor>(device, std::move(shape), dtype);

            switch (dtype) {
                case core::DType::float32:
                    fill_uniform<float>(*tensor, generator, bound);
                    break;
                case core::DType::float64:
                    fill_uniform<double>(*tensor, generator, bound);
                    break;
                case core::DType::float16:
                    fill_uniform<core::Float16>(*tensor, generator, bound);
                    break;
                default:
                    ORMI_THROW(DTypeError, "Linear parameters require a floating-point dtype.");
            }

            return tensor;
        }
    }

    Linear::Linear(
        std::int64_t in_features,
        std::int64_t out_features,
        bool bias,
        core::DType dtype,
        core::Device device,
        std::optional<std::uint64_t> seed
    ) {
        if (in_features <= 0 || out_features <= 0) {
            ORMI_THROW(ShapeError, "Linear feature counts must be positive.");
        }

        if (!core::is_floating(dtype)) {
            ORMI_THROW(DTypeError, "Linear parameters require a floating-point dtype.");
        }

        std::mt19937_64 generator(seed ? *seed : std::random_device{}());
        const auto bound = 1.0 / std::sqrt(static_cast<double>(in_features));

        weight_ = create_parameter({in_features, out_features}, dtype, device, generator, bound);
        register_parameter(weight_, "weight");

        if (!bias) {
            return;
        }

        bias_ = create_parameter({out_features}, dtype, device, generator, bound);
        register_parameter(bias_, "bias");
    }

    Linear::Linear(std::shared_ptr<core::Tensor> weight, std::shared_ptr<core::Tensor> bias)
        : weight_(std::move(weight)),
          bias_(std::move(bias)) {
        if (!weight_ || weight_->rank() != 2 || weight_->shape()[0] <= 0 || weight_->shape()[1] <= 0) {
            ORMI_THROW(
                ShapeError,
                "Linear weight must have shape [in_features, out_features] with positive axis sizes."
            );
        }

        if (!bias_) {
            register_parameter(weight_, "weight");
            return;
        }

        if (bias_->shape() != core::Shape{out_features()}) {
            ORMI_THROW(ShapeError, "Linear bias must have shape [out_features].");
        }

        if (bias_->dtype() != weight_->dtype()) {
            ORMI_THROW(DTypeError, "Linear weight and bias dtypes must match.");
        }

        if (bias_->device() != weight_->device()) {
            ORMI_THROW(DeviceError, "Linear weight and bias devices must match.");
        }

        register_parameter(weight_, "weight");
        register_parameter(bias_, "bias");
    }

    std::shared_ptr<core::Tensor> Linear::forward(const std::shared_ptr<core::Tensor>& input) const {
        if (!input) {
            ORMI_THROW(OperationError, "Linear requires an input tensor.");
        }

        if (input->rank() == 0 || input->shape()[input->rank() - 1] != in_features()) {
            ORMI_THROW(ShapeError, "Linear input's last dimension must equal in_features.");
        }

        auto output = core::ops::matmul(input, weight_);
        return bias_ ? core::ops::add(output, bias_) : output;
    }

    std::int64_t Linear::in_features() const noexcept {
        return weight_->shape()[0];
    }

    std::int64_t Linear::out_features() const noexcept {
        return weight_->shape()[1];
    }

    std::shared_ptr<core::Tensor> Linear::weight() const noexcept {
        return weight_;
    }

    std::shared_ptr<core::Tensor> Linear::bias() const noexcept {
        return bias_;
    }
}
