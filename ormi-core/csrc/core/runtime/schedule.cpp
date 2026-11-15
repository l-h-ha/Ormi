#include <cstddef>
#include <utility>
#include <vector>

#include <ormi/core/tensor.h>

#include "core/runtime/device.h"
#include "core/runtime/fusion/group.h"
#include "core/runtime/lower.h"
#include "core/runtime/schedule.h"

namespace ormi::core::runtime::schedule {
    namespace {
        struct Frame {
            explicit Frame(Step value) : step(std::move(value)) {}

            Step step;
            std::size_t next = 0;
        };

        template <typename Complete, typename Prepare, typename Finish>
        void walk(
            Step root_step,
            Complete complete,
            Prepare prepare_step,
            Finish finish
        ) {
            std::vector<Frame> stack;
            stack.reserve(16);
            stack.emplace_back(std::move(root_step));

            while (!stack.empty()) {
                auto& frame = stack.back();
                bool descended = false;

                while (frame.next < frame.step.input_count()) {
                    auto& input = frame.step.input(frame.next++);

                    if (complete(input)) {
                        continue;
                    }

                    stack.emplace_back(prepare_step(input));
                    descended = true;
                    break;
                }

                if (descended) {
                    continue;
                }

                auto step = std::move(frame.step);
                stack.pop_back();
                finish(std::move(step));
            }
        }

        template <typename Complete, typename Prepare, typename Finish>
        void walk(
            Tensor& root,
            Complete complete,
            Prepare prepare_step,
            Finish finish
        ) {
            if (complete(root)) {
                return;
            }

            auto root_step = prepare_step(root);
            walk(
                std::move(root_step),
                std::move(complete),
                std::move(prepare_step),
                std::move(finish)
            );
        }
    }

    void run(Tensor& root) {
        if (root.realized()) {
            return;
        }

        auto root_step = lower(root, Preparation::direct);
        std::vector<Tensor*> dependencies;

        for (std::size_t i = 0; i < root_step.input_count(); ++i) {
            auto& input = root_step.input(i);

            if (!input.realized()) {
                if (dependencies.empty()) {
                    dependencies.reserve(root_step.input_count());
                }

                dependencies.push_back(&input);
            }
        }

        if (dependencies.empty()) {
            root_step.run();
            return;
        }

        std::unordered_set<const Tensor*> shared_scalars;
        const bool scalar = root.numel() == 1 && root.device().is_host();
        const auto minimum_nodes = scalar ? backend(root.device()).fusion.max_operations : 0;
        auto* root_tensor = &root;
        const auto roots = scalar
            ? std::span<Tensor* const>{&root_tensor, 1}
            : std::span<Tensor* const>{dependencies};
        const auto candidates = fusion::find_candidates(
            roots, scalar ? &shared_scalars : nullptr, minimum_nodes
        );

        if (!shared_scalars.empty()) {
            root_step = lower(root, candidates, shared_scalars, Preparation::direct);
        }

        walk(
            std::move(root_step),
            [](const Tensor& tensor) {
                return tensor.realized();
            },
            [&candidates, &shared_scalars](Tensor& tensor) {
                return shared_scalars.empty()
                    ? lower(tensor, candidates, Preparation::direct)
                    : lower(tensor, candidates, shared_scalars, Preparation::direct);
            },
            [](Step step) {
                step.run();
            }
        );
    }

    void append(
        Tensor& root,
        std::unordered_set<const Tensor*>& completed,
        std::vector<Step>& steps,
        const fusion::Candidates& candidates,
        std::unordered_set<const Tensor*>* boundaries
    ) {
        walk(
            root,
            [&completed](const Tensor& tensor) {
                return
                    tensor.realized() ||
                    completed.contains(&tensor);
            },
            [&completed, &candidates, boundaries](Tensor& tensor) {
                return lower(tensor, candidates, boundaries ? *boundaries : completed, Preparation::deferred);
            },
            [&completed, &steps, boundaries](Step step) {
                for (std::size_t i = 0; i < step.output_count(); ++i) {
                    completed.insert(&step.output(i));

                    if (boundaries) {
                        boundaries->insert(&step.output(i));
                    }
                }

                steps.push_back(std::move(step));
            }
        );
    }
}
