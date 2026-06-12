#pragma once
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <vector>
#include <memory>
#include "core/ops.h"
#include "core/attributes.h"

namespace ormi {
    struct Node;

    // Signatures
    using ShapeInferFunc = std::function<std::vector<int>(const std::vector<std::shared_ptr<Node>>&, const OpAttributes&)>;
    using StrideInferFunc = std::function<std::vector<int>(const std::vector<std::shared_ptr<Node>>&, const OpAttributes&, const std::vector<int>& out_shape)>;
    using AdjointFunc = std::function<void(std::shared_ptr<Node>)>;

    // Templates
    template<typename FuncType>
    class Registry {
    public:
        static std::unordered_map<ops::OpEnum, FuncType>& get_map() {
            static std::unordered_map<ops::OpEnum, FuncType> map;
            return map;
        }

        static void add(ops::OpEnum op, FuncType func) {
            get_map()[op] = func;
        }

        static FuncType get(ops::OpEnum op) {
            auto& map = get_map();
            auto it = map.find(op);
            if (it != map.end()) return it->second;
            return nullptr;
        }
    };

    class ViewOpRegistry {
    public:
        static ViewOpRegistry& get() {
            static ViewOpRegistry instance;
            return instance;
        }
        void add(ops::OpEnum op) { ops_.insert(op); }
        bool contains(ops::OpEnum op) const { return ops_.count(op) > 0; }
    private:
        std::unordered_set<ops::OpEnum> ops_;
    };

    //
    struct RegisterViewOp {
        RegisterViewOp(ops::OpEnum op) {
            ViewOpRegistry::get().add(op);
        }
    };
    struct RegisterShape {
        RegisterShape(ops::OpEnum op, ShapeInferFunc func) { Registry<ShapeInferFunc>::add(op, func); }
    };
    struct RegisterStride {
        RegisterStride(ops::OpEnum op, StrideInferFunc func) { Registry<StrideInferFunc>::add(op, func); }
    };
    struct RegisterAdjoint {
        RegisterAdjoint(ops::OpEnum op, AdjointFunc func) { Registry<AdjointFunc>::add(op, func); }
    };
}