#pragma once
#include <cstddef>
#include <memory>
#include <vector>

namespace ormi::core {
    class Tensor;

    // A fixed sequence of steps that reads and writes the original tensors directly.
    // Runs and replays that share tensor storage must not overlap.
    class Plan final {
        class Impl;

    public:
        // Retains the Plan's tensors and workspace, even after the Plan is destroyed.
        class Replay final {
        public:
            ~Replay();

            Replay(const Replay&) = delete;
            Replay& operator=(const Replay&) = delete;
            Replay(Replay&&) noexcept;
            Replay& operator=(Replay&&) noexcept;

            // Run the recording without rebuilding or walking the tensor graph.
            void run();

        private:
            class Impl;

            explicit Replay(std::unique_ptr<Impl> impl);

            std::unique_ptr<Impl> impl_;

            friend class Plan::Impl;
        };

        // Copy value into destination after all scheduled steps have finished submitting work.
        // Both must be contiguous and match shape, dtype, and device. Destination must be realized.
        struct Update {
            std::shared_ptr<Tensor> destination;
            std::shared_ptr<Tensor> value;
        };

        // Outputs must be lazy. Already-realized dependencies are treated as inputs.
        // Construction prepares steps and may compile kernels. Call run() to execute them.
        explicit Plan(
            std::vector<std::shared_ptr<Tensor>> outputs,
            std::vector<Update> updates = {}
        );
        ~Plan();

        Plan(const Plan&) = delete;
        Plan& operator=(const Plan&) = delete;
        Plan(Plan&&) noexcept;
        Plan& operator=(Plan&&) noexcept;

        // Rerun all steps and explicit updates, reusing output storage and reading current leaf values.
        // CUDA work may still be queued when this call returns.
        void run();
        // Requires a successful run, one device, and backend recording support.
        Replay record();
        std::size_t steps() const noexcept;
        // Count only step output allocations and reported setup workspace.
        std::size_t storage_buffers() const noexcept;
        std::size_t storage_bytes() const noexcept;

    private:
        std::shared_ptr<Impl> impl_;
    };
}
