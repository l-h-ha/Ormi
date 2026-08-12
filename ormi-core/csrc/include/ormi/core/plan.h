#pragma once
#include <cstddef>
#include <memory>
#include <vector>

namespace ormi::core {
    class Tensor;

    class Plan final {
        class Impl;

    public:
        class Replay final {
        public:
            ~Replay();

            Replay(const Replay&) = delete;
            Replay& operator=(const Replay&) = delete;
            Replay(Replay&&) noexcept;
            Replay& operator=(Replay&&) noexcept;

            void run();

        private:
            class Impl;

            explicit Replay(std::unique_ptr<Impl> impl);

            std::unique_ptr<Impl> impl_;

            friend class Plan::Impl;
        };

        struct Update {
            std::shared_ptr<Tensor> destination;
            std::shared_ptr<Tensor> value;
        };

        explicit Plan(
            std::vector<std::shared_ptr<Tensor>> outputs,
            std::vector<Update> updates = {}
        );
        ~Plan();

        Plan(const Plan&) = delete;
        Plan& operator=(const Plan&) = delete;
        Plan(Plan&&) noexcept;
        Plan& operator=(Plan&&) noexcept;

        void run();
        Replay record();
        std::size_t steps() const noexcept;
        std::size_t storage_buffers() const noexcept;
        std::size_t storage_bytes() const noexcept;

    private:
        std::shared_ptr<Impl> impl_;
    };
}
