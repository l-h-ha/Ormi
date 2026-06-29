#pragma once
#include <functional>
#include <memory>

namespace ormi {
    class CudaGraphTape {
    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl;

    public:
        CudaGraphTape();
        ~CudaGraphTape();

        void capture(std::function<void()> step_fn);
        void execute();

        size_t node_count() const;
    };
}