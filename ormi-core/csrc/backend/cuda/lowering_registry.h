#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <stdexcept>
#include "core/attributes.h"
#include "core/ops.h"

namespace ormi::cuda {
    using LoweringFunc = std::function<std::string(const std::vector<std::string>&)>;

    class CUDALoweringRegistry {
    public:
        static void register_func(ops::OpEnum op, LoweringFunc func) {
            get_registry()[op] = func;
        }

        static LoweringFunc get(ops::OpEnum op) {
            auto& reg = get_registry();
            if (reg.find(op) == reg.end()) {
                throw std::runtime_error("CUDA JIT Compiler Error: No lowering rule registered for this operation.");
            }
            return reg[op];
        }

    private:
        static std::unordered_map<ops::OpEnum, LoweringFunc>& get_registry() {
            static std::unordered_map<ops::OpEnum, LoweringFunc> registry;
            return registry;
        }
    };

    struct RegisterCudaLowering {
        RegisterCudaLowering(ops::OpEnum op, LoweringFunc func) {
            CUDALoweringRegistry::register_func(op, func);
        }
    };
}