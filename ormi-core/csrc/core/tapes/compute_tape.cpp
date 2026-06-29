#include "compute_tape.h"

namespace ormi {
    void ComputeTape::capture(const std::vector<std::shared_ptr<Node>>& roots) {
        auto sorted = Topology::sort(roots);
        execution_order.clear();

        for (auto& node : sorted) {
            if (!node) {
                continue;
            }

            if (node->op != ops::OpEnum::LEAF) {
                execution_order.push_back(node);
            }
        }
    }

    void ComputeTape::execute() {
        for (auto& node : execution_order) {
            execute_op(node); 
        }
    }

    void ComputeTape::dump() const {
        std::cout << "ComputeTape execution order:\n";

        for (const auto& node : execution_order) {
            std::cout << "  "
                    << "op=" << static_cast<int>(node->op)
                    << " name=" << node->name
                    << " dtype=" << node->dtype
                    << " data=" << node->data.get()
                    << "\n";
        }
    }
}