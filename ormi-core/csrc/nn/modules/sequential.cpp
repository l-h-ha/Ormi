#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include <ormi/core/error.h>
#include <ormi/nn/modules/sequential.h>

namespace ormi::nn {
    Sequential::Sequential(std::vector<std::shared_ptr<Module>> layers) {
        for (auto& layer : layers) {
            register_module(std::move(layer));
        }
    }

    void Sequential::append(std::shared_ptr<Module> layer) {
        register_module(std::move(layer));
    }

    std::shared_ptr<core::Tensor> Sequential::forward(const std::shared_ptr<core::Tensor>& input) const {
        if (!input) {
            ORMI_THROW(OperationError, "Sequential requires an input tensor.");
        }

        auto output = input;
        const auto& layers = children();
        const auto count = layers.size();

        for (std::size_t index = 0; index < count; ++index) {
            output = layers[index]->forward(output);
        }

        return output;
    }
}
