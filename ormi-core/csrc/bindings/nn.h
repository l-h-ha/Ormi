#pragma once
#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/trampoline_self_life_support.h>

#include <ormi/core/error.h>
#include <ormi/core/tensor.h>
#include <ormi/nn/modules/module.h>

namespace ormi::bindings {
    class PyModule : public nn::Module, public pybind11::trampoline_self_life_support {
    public:
        using nn::Module::Module;
        using nn::Module::register_module;
        using nn::Module::register_parameter;
        using nn::Module::register_buffer;

        std::shared_ptr<core::Tensor> forward(const std::shared_ptr<core::Tensor>& input) const override {
            if (!input) {
                ORMI_THROW(OperationError, "Module requires an input tensor.");
            }

            pybind11::gil_scoped_acquire gil;
            const auto forward = pybind11::get_override(this, "forward");
            if (!forward) {
                ORMI_THROW(OperationError, "A Python Module subclass must implement forward().");
            }

            const auto output = forward(input);
            if (!pybind11::isinstance<core::Tensor>(output)) {
                ORMI_THROW(OperationError, "Module.forward() must return a Tensor.");
            }

            return output.cast<std::shared_ptr<core::Tensor>>();
        }
    };
}
