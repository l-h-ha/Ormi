#pragma once

#include <memory>
#include <vector>

#include "core/node.h"

namespace ormi::optim {
    class SGD {
    private:
        struct ParamGrad {
            std::shared_ptr<Node> param;
            std::shared_ptr<Node> grad;
        };

        std::vector<ParamGrad> params;
        float lr;

    public:
        SGD(
            const std::vector<std::pair<std::shared_ptr<Node>, 
            std::shared_ptr<Node>>>& parameter_grads,
            float learning_rate
        );

        void step();
    };
}