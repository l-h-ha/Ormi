#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

namespace ormi::core {
    class Tensor;
}

namespace ormi::core::optim {
    struct ScalePolicy {
        double growth = 2.0;
        double backoff = 0.5;
        std::size_t interval = 2000;
        double limit = std::numeric_limits<double>::max();
    };

    class GradScaler {
    public:
        explicit GradScaler(double scale);
        GradScaler(double scale, ScalePolicy policy);

        double value() const noexcept;

        std::shared_ptr<Tensor> scale(const std::shared_ptr<Tensor>& loss) const;
        std::optional<std::shared_ptr<Tensor>> step(
            const std::shared_ptr<Tensor>& parameter,
            const std::shared_ptr<Tensor>& scaled_gradient,
            double learning_rate
        );
        std::optional<std::vector<std::shared_ptr<Tensor>>> step(
            const std::vector<std::shared_ptr<Tensor>>& parameters,
            const std::vector<std::shared_ptr<Tensor>>& scaled_gradients,
            double learning_rate
        );

    private:
        void update(bool finite) noexcept;

        double scale_;
        ScalePolicy policy_;
        std::size_t finite_steps_;
    };
}
