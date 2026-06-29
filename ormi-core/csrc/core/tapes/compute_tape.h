#pragma once
#include <cstring>
#include "backend/runtime.h"
#include "tape.h"

namespace ormi {
    class ComputeTape : public Tape {
    private:
        std::vector<std::shared_ptr<Node>> execution_order;

    public:
        void capture(const std::vector<std::shared_ptr<Node>>& roots);
        void execute() override;
        void dump() const;
    };
}