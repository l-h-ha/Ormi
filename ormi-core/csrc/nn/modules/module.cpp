#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/nn/modules/module.h>

namespace ormi::nn {
    std::string Module::registration_name(std::string name, std::string fallback) const {
        if (name.empty()) {
            name = std::move(fallback);
        }
        if (name.find_first_of(".\\/") != std::string::npos || name.find('\0') != std::string::npos) {
            ORMI_THROW(OperationError, "Module names cannot contain '.', '/', '\\\\', or null bytes.");
        }
        for (const auto& entry : parameters_) {
            if (entry.first == name) {
                ORMI_THROW(OperationError, "A module registration name is already in use: " + name);
            }
        }
        for (const auto& entry : buffers_) {
            if (entry.first == name) {
                ORMI_THROW(OperationError, "A module registration name is already in use: " + name);
            }
        }
        for (const auto& child_name : child_names_) {
            if (child_name == name) {
                ORMI_THROW(OperationError, "A module registration name is already in use: " + name);
            }
        }
        return name;
    }

    void Module::register_parameter(std::shared_ptr<core::Tensor> parameter, std::string name) {
        if (!parameter) {
            ORMI_THROW(OperationError, "A module parameter cannot be null.");
        }

        if (parameter->op() || !parameter->realized()) {
            ORMI_THROW(OperationError, "A module parameter must be a realized leaf tensor.");
        }

        if (!core::is_floating(parameter->dtype())) {
            ORMI_THROW(DTypeError, "A module parameter must have a floating-point dtype.");
        }

        name = registration_name(std::move(name), "parameter_" + std::to_string(parameters_.size()));
        parameters_.emplace_back(std::move(name), std::move(parameter));
    }

    void Module::register_module(std::shared_ptr<Module> module, std::string name) {
        if (!module) {
            ORMI_THROW(OperationError, "A child module cannot be null.");
        }

        std::vector<const Module*> pending{module.get()};
        std::unordered_set<const Module*> visited;

        while (!pending.empty()) {
            const auto* current = pending.back();
            pending.pop_back();

            if (current == this) {
                ORMI_THROW(OperationError, "Child modules cannot form an ownership cycle.");
            }

            if (!visited.insert(current).second) {
                continue;
            }

            for (const auto& child : current->children_) {
                pending.push_back(child.get());
            }
        }

        name = registration_name(std::move(name), std::to_string(children_.size()));
        child_names_.push_back(std::move(name));
        try {
            children_.push_back(std::move(module));
        } catch (...) {
            child_names_.pop_back();
            throw;
        }
    }

    void Module::register_buffer(std::shared_ptr<core::Tensor> buffer, std::string name) {
        if (!buffer || buffer->op() || !buffer->realized()) {
            ORMI_THROW(OperationError, "A module buffer must be a realized leaf tensor.");
        }
        name = registration_name(std::move(name), "buffer_" + std::to_string(buffers_.size()));
        buffers_.emplace_back(std::move(name), std::move(buffer));
    }

    std::vector<std::shared_ptr<core::Tensor>> Module::buffers() const {
        std::vector<std::shared_ptr<core::Tensor>> result;
        std::unordered_set<const core::Tensor*> visited_buffers;
        std::unordered_set<const Module*> visited_modules;
        std::vector<const Module*> pending{this};

        while (!pending.empty()) {
            const auto* current = pending.back();
            pending.pop_back();
            if (!visited_modules.insert(current).second) {
                continue;
            }
            for (const auto& entry : current->buffers_) {
                const auto& buffer = entry.second;
                if (visited_buffers.insert(buffer.get()).second) {
                    result.push_back(buffer);
                }
            }
            for (auto child = current->children_.rbegin(); child != current->children_.rend(); ++child) {
                pending.push_back(child->get());
            }
        }
        return result;
    }

    void Module::train(bool mode) {
        std::vector<Module*> pending{this};
        std::unordered_set<Module*> visited;
        while (!pending.empty()) {
            auto* current = pending.back();
            pending.pop_back();
            if (!visited.insert(current).second) {
                continue;
            }
            current->training_ = mode;
            for (const auto& child : current->children_) {
                pending.push_back(child.get());
            }
        }
    }

    void Module::eval() {
        train(false);
    }

    bool Module::training() const noexcept {
        return training_;
    }

    std::vector<std::shared_ptr<core::Tensor>> Module::parameters() const {
        std::vector<std::shared_ptr<core::Tensor>> result;
        std::unordered_set<const core::Tensor*> visited_parameters;
        std::unordered_set<const Module*> visited_modules;
        std::vector<const Module*> pending{this};

        while (!pending.empty()) {
            const auto* current = pending.back();
            pending.pop_back();

            if (!visited_modules.insert(current).second) {
                continue;
            }

            for (const auto& entry : current->parameters_) {
                const auto& parameter = entry.second;
                if (visited_parameters.insert(parameter.get()).second) {
                    result.push_back(parameter);
                }
            }

            // Reverse the push order so the first registered child is visited first.
            for (auto child = current->children_.rbegin(); child != current->children_.rend(); ++child) {
                pending.push_back(child->get());
            }
        }

        return result;
    }

    const std::vector<std::shared_ptr<Module>>& Module::children() const noexcept {
        return children_;
    }

    std::map<std::string, std::shared_ptr<core::Tensor>> Module::state() const {
        std::map<std::string, std::shared_ptr<core::Tensor>> result;
        std::vector<std::pair<std::string, const Module*>> pending{{"", this}};
        while (!pending.empty()) {
            auto [prefix, current] = std::move(pending.back());
            pending.pop_back();
            for (const auto& [name, tensor] : current->parameters_) {
                result.emplace(prefix + name, tensor);
            }
            for (const auto& [name, tensor] : current->buffers_) {
                result.emplace(prefix + name, tensor);
            }
            for (std::size_t i = current->children_.size(); i > 0; --i) {
                pending.emplace_back(
                    prefix + current->child_names_[i - 1] + ".",
                    current->children_[i - 1].get()
                );
            }
        }
        return result;
    }
}
