#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include "core/attributes.h"
#include "core/ops.h"
#include "core/error.h"

namespace ormi::cpu {
    using LoweringFunc = std::string(*)(const std::vector<std::string>&);

    class CPULoweringRegistry {
    public:
        static void register_func(ops::OpEnum op, LoweringFunc func) {
            get_registry()[op] = func;
        }

        static LoweringFunc get(ops::OpEnum op) {
            auto& reg = get_registry();
            if (reg.find(op) == reg.end()) {
                ORMI_THROW(RegistryError, "Unrecognized operation registry.");
            }
            return reg[op];
        }

    private:
        static std::unordered_map<ops::OpEnum, LoweringFunc>& get_registry() {
            static std::unordered_map<ops::OpEnum, LoweringFunc> registry;
            return registry;
        }
    };

    struct RegisterCPULowering {
        RegisterCPULowering(ops::OpEnum op, LoweringFunc func) {
            CPULoweringRegistry::register_func(op, func);
        }
    };
}