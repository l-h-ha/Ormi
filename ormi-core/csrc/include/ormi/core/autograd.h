#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace ormi::core {
    // Collects lazy gradients for each Tensor object, adding repeated contributions.
    class Grads {
    public:
        void add(const std::shared_ptr<Tensor>& tensor, std::shared_ptr<Tensor> gradient);
        std::shared_ptr<Tensor> get(const std::shared_ptr<Tensor>& tensor) const;

    private:
        std::unordered_map<const Tensor*, std::shared_ptr<Tensor>> values_;
    };

    // Sum axes introduced or expanded by broadcasting, then restore the input shape.
    std::shared_ptr<Tensor> unbroadcast(
        const std::shared_ptr<Tensor>& gradient,
        const Shape& shape
    );

    // Build gradients without computing output values. Results can be differentiated again.
    // A scalar output defaults to seed 1. Other outputs need a seed with matching
    // shape, dtype, and device. Inputs with no contribution receive a lazy zero tensor.
    std::vector<std::shared_ptr<Tensor>> grad(
        const std::shared_ptr<Tensor>& output,
        const std::vector<std::shared_ptr<Tensor>>& inputs,
        std::shared_ptr<Tensor> seed = nullptr
    );
    std::shared_ptr<Tensor> grad(
        const std::shared_ptr<Tensor>& output,
        const std::shared_ptr<Tensor>& input,
        std::shared_ptr<Tensor> seed = nullptr
    );
    // Repeated scalar differentiation. Each intermediate result must satisfy grad's seed rule.
    // Order zero returns output unchanged.
    std::shared_ptr<Tensor> diff(
        const std::shared_ptr<Tensor>& output,
        const std::shared_ptr<Tensor>& input,
        std::size_t order
    );

    // For matching point dimensions P, output is (*P, components), input is
    // (*P, coordinates), and the result is (*P, 1). Each entry differentiates
    // its own point, even when other points affect that output. Orders are positive.
    // Omit coordinate/component only when its axis has size 1. Negative indices count from the end.
    std::shared_ptr<Tensor> partial(
        const std::shared_ptr<Tensor>& output,
        const std::shared_ptr<Tensor>& input,
        std::optional<std::int64_t> coordinate = std::nullopt,
        std::optional<std::int64_t> component = std::nullopt,
        std::size_t order = 1
    );
}
