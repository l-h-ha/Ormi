#pragma once
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include "core/attributes.h"
#include "core/ops.h"

namespace ormi::cpu {
    using LoweringFunc = void(*)(std::vector<float>& regs, int dest, int src1, int src2);

    class CPULoweringRegistry {
    public:
        static void register_func(ops::OpEnum op, LoweringFunc func) {
            get_registry()[op] = func;
        }

        static LoweringFunc get(ops::OpEnum op) {
            auto& reg = get_registry();
            if (reg.find(op) == reg.end()) {
                throw std::runtime_error("CPU JIT Compiler Error: No lowering rule registered for this operation.");
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