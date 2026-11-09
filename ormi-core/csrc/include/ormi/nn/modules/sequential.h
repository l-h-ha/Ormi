#pragma once
#include <memory>
#include <vector>

#include <ormi/core/tensor.h>
#include <ormi/nn/modules/module.h>

namespace ormi::nn {
    // Calls registered children in order to build one graph. An empty sequence returns its input.
    class Sequential final : public Module {
    public:
        explicit Sequential(std::vector<std::shared_ptr<Module>> layers = {});

        // Appended layers run on subsequent forward calls. Existing graphs keep their original nodes.
        void append(std::shared_ptr<Module> layer);
        std::shared_ptr<core::Tensor> forward(const std::shared_ptr<core::Tensor>& input) const override;
    };
}
