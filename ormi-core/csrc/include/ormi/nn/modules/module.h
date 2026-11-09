#pragma once
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ormi/core/tensor.h>

namespace ormi::nn {
    // Owns registered parameters, buffers, and child modules. forward builds tensor graphs.
    // Prepared execution runs their tensor operations directly.
    class Module {
    public:
        Module() = default;
        virtual ~Module() = default;

        Module(const Module&) = delete;
        Module& operator=(const Module&) = delete;

        // Build a lazy graph without computing input or output values.
        virtual std::shared_ptr<core::Tensor> forward(const std::shared_ptr<core::Tensor>& input) const = 0;

        std::shared_ptr<core::Tensor> operator()(const std::shared_ptr<core::Tensor>& input) const {
            return forward(input);
        }

        // List this module's parameters, then each child's recursively. Shared tensors appear once.
        std::vector<std::shared_ptr<core::Tensor>> parameters() const;
        // Non-trainable state, such as running statistics, in the same traversal order as parameters().
        std::vector<std::shared_ptr<core::Tensor>> buffers() const;
        // Live parameters and buffers by path, including every path to shared state.
        // This does not copy data or execute graphs.
        std::map<std::string, std::shared_ptr<core::Tensor>> state() const;
        const std::vector<std::shared_ptr<Module>>& children() const noexcept;
        // Set this module and its registered children recursively. Mode affects future forward calls.
        // Existing graphs and Plans keep the behavior selected when they were built.
        void train(bool mode = true);
        void eval();
        bool training() const noexcept;

    protected:
        // Names are local and unique across parameters, buffers, and children.
        // Names cannot contain a period, slash, backslash, or null byte.
        // Empty names use parameter_N, buffer_N, or the child index. Prefer names for custom state.
        // Registration retains ownership. Parameters must be floating-point, realized leaves.
        void register_parameter(std::shared_ptr<core::Tensor> parameter, std::string name = {});
        // Buffers must be realized leaves of any supported dtype.
        void register_buffer(std::shared_ptr<core::Tensor> buffer, std::string name = {});
        // Keeps the child alive and rejects ownership cycles. C++ member fields are not auto-registered.
        void register_module(std::shared_ptr<Module> module, std::string name = {});

    private:
        std::string registration_name(std::string name, std::string fallback) const;

        std::vector<std::pair<std::string, std::shared_ptr<core::Tensor>>> parameters_;
        std::vector<std::pair<std::string, std::shared_ptr<core::Tensor>>> buffers_;
        std::vector<std::shared_ptr<Module>> children_;
        // Uses the same indices as children_. Forward execution reads children_ directly.
        std::vector<std::string> child_names_;
        bool training_ = true;
    };
}
