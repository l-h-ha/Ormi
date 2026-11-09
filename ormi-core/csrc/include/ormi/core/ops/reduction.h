#pragma once
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <vector>

#include <ormi/core/tensor.h>

namespace ormi::core::ops {
    std::shared_ptr<Tensor> all_finite(const std::shared_ptr<Tensor>& input);
    std::shared_ptr<Tensor> max(
        const std::shared_ptr<Tensor>& input,
        std::vector<std::int64_t> axes,
        bool keepdims = false
    );

    std::shared_ptr<Tensor> sum(
        const std::shared_ptr<Tensor>& input,
        std::vector<std::int64_t> axes,
        bool keepdims = false
    );
    std::shared_ptr<Tensor> sum(
        const std::shared_ptr<Tensor>& input,
        std::initializer_list<std::int64_t> axes,
        bool keepdims = false
    );
    std::shared_ptr<Tensor> sum(const std::shared_ptr<Tensor>& input, bool keepdims = false);

    std::shared_ptr<Tensor> mean(
        const std::shared_ptr<Tensor>& input,
        std::vector<std::int64_t> axes,
        bool keepdims = false
    );
    std::shared_ptr<Tensor> mean(
        const std::shared_ptr<Tensor>& input,
        std::initializer_list<std::int64_t> axes,
        bool keepdims = false
    );
    std::shared_ptr<Tensor> mean(const std::shared_ptr<Tensor>& input, bool keepdims = false);

    bool is_sum(const Tensor& tensor);
    bool is_mean(const Tensor& tensor);
}
