#pragma once
#include <memory>
#include <vector>

#include <ormi/core/tensor.h>
#include <ormi/nn/modules/module.h>

namespace ormi::nn {
    // Build a graph by calling children in order. Empty sequences return input.
    class Sequential final : public Module {
    public:
        explicit Sequential(std::vector<std::shared_ptr<Module>> layers = {});

        // Append a layer for future forward calls.
        void append(std::shared_ptr<Module> layer);
        std::shared_ptr<core::Tensor> forward(const std::shared_ptr<core::Tensor>& input) const override;
    };
}
