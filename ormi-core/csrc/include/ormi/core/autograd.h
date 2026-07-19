#pragma once
#include <cstddef>
#include <memory>
#include <unordered_map>
#include <vector>

#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace ormi::core {
    class Grads {
    public:
        void add(const std::shared_ptr<Tensor>& tensor, std::shared_ptr<Tensor> gradient);
        std::shared_ptr<Tensor> get(const std::shared_ptr<Tensor>& tensor) const;

    private:
        std::unordered_map<const Tensor*, std::shared_ptr<Tensor>> values_;
    };

    std::shared_ptr<Tensor> unbroadcast(
        const std::shared_ptr<Tensor>& gradient,
        const Shape& shape
    );

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
    std::shared_ptr<Tensor> diff(
        const std::shared_ptr<Tensor>& output,
        const std::shared_ptr<Tensor>& input,
        std::size_t order
    );
}
