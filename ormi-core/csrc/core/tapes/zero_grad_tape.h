#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "core/device.h"
#include "core/node.h"
#include "tape.h"

namespace ormi {
    class ZeroGradTape : public Tape {
    private:
        struct MemBlock { 
            void* ptr; 
            size_t bytes;
            Device device;
        };
        std::vector<MemBlock> blocks;

    public:
        void capture(const std::vector<std::shared_ptr<Node>>& roots);
        void execute() override; 
    };
}