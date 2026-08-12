#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda.h>
#include <nvrtc.h>

#include <ormi/core/error.h>

#include "backends/cuda/check.cuh"
#include "backends/cuda/fusion/cache.h"
#include "backends/cuda/fusion/codegen.h"
#include "backends/cuda/fusion/compile.h"
#include "backends/cuda/fusion/tile.h"
#include "backends/cuda/state.h"
#include "core/runtime/fusion/key.h"

namespace ormi::backends::cuda::fusion {
    namespace {
        class Program final {
        public:
            explicit Program(const std::string& source) {
                const auto status = nvrtcCreateProgram(
                    &program_,
                    source.c_str(),
                    "ormi_fusion.cu",
                    0,
                    nullptr,
                    nullptr
                );

                if (status != NVRTC_SUCCESS) {
                    ORMI_THROW(
                        jitError,
                        std::string("nvrtcCreateProgram failed: ") +
                            nvrtcGetErrorString(status)
                    );
                }
            }

            ~Program() noexcept {
                if (program_) {
                    (void)nvrtcDestroyProgram(&program_);
                }
            }

            Program(const Program&) = delete;
            Program& operator=(const Program&) = delete;

            std::string compile(int major, int minor) {
                const auto architecture =
                    "--gpu-architecture=compute_" +
                    std::to_string(major) + std::to_string(minor);
                const char* options[]{"--std=c++17", architecture.c_str()};
                const auto status = nvrtcCompileProgram(program_, 2, options);

                std::size_t log_size = 0;
                (void)nvrtcGetProgramLogSize(program_, &log_size);
                std::string log(log_size, '\0');
                if (log_size != 0) {
                    (void)nvrtcGetProgramLog(program_, log.data());
                }

                if (status != NVRTC_SUCCESS) {
                    ORMI_THROW(
                        jitError,
                        std::string("CUDA fusion compilation failed: ") +
                            nvrtcGetErrorString(status) + "\n" + log
                    );
                }

                std::size_t ptx_size = 0;
                if (nvrtcGetPTXSize(program_, &ptx_size) != NVRTC_SUCCESS) {
                    ORMI_THROW(
                        jitError,
                        "Cannot determine CUDA fusion PTX size."
                    );
                }

                std::string ptx(ptx_size, '\0');
                if (nvrtcGetPTX(program_, ptx.data()) != NVRTC_SUCCESS) {
                    ORMI_THROW(jitError, "Cannot retrieve CUDA fusion PTX.");
                }
                return ptx;
            }

        private:
            nvrtcProgram program_ = nullptr;
        };

        std::string context(int major, int minor) {
            int nvrtc_major = 0;
            int nvrtc_minor = 0;
            const auto status = nvrtcVersion(&nvrtc_major, &nvrtc_minor);
            if (status != NVRTC_SUCCESS) {
                ORMI_THROW(
                    jitError,
                    std::string("nvrtcVersion failed: ") +
                        nvrtcGetErrorString(status)
                );
            }
            return
                "compute_" + std::to_string(major) +
                std::to_string(minor) +
                "\nnvrtc_" + std::to_string(nvrtc_major) +
                "." + std::to_string(nvrtc_minor);
        }

        std::uint64_t combine(std::uint64_t hash, std::uint64_t value) {
            hash ^= value;
            hash *= 1099511628211ull;
            return hash;
        }

        struct Entry {
            Entry(
                int selected_device,
                int selected_major,
                int selected_minor,
                const core::runtime::fusion::Expr& expression,
                std::shared_ptr<Compiled> compiled_kernel
            )
                : device(selected_device),
                  major(selected_major),
                  minor(selected_minor),
                  key(expression),
                  compiled(std::move(compiled_kernel)) {}

            int device;
            int major;
            int minor;
            core::runtime::fusion::Key key;
            std::shared_ptr<Compiled> compiled;

            bool matches(
                int requested_device,
                int requested_major,
                int requested_minor,
                const core::runtime::fusion::Expr& expression
            ) const {
                return
                    device == requested_device &&
                    major == requested_major &&
                    minor == requested_minor &&
                    key.matches(expression);
            }
        };

        std::shared_ptr<Compiled> build(
            const core::runtime::fusion::Expr& expression,
            const State& cuda_state,
            int major,
            int minor
        ) {
            auto tiled = tile(expression);
            const auto source = emit(expression);
            const auto artifact = cache::get(
                source,
                context(major, minor),
                [&] {
                    Program program(source);
                    return program.compile(major, minor);
                }
            );
            auto result = std::make_shared<Compiled>();
            result->tiled = std::move(tiled);

            check_driver(
                cuModuleLoadData(&result->module, artifact.ptx.data()),
                "cuModuleLoadData"
            );

            try {
                check_driver(
                    cuModuleGetFunction(
                        &result->function,
                        result->module,
                        "ormi_kernel"
                    ),
                    "cuModuleGetFunction"
                );

                if (expression.reduction && !result->tiled) {
                    check_driver(
                        cuModuleGetFunction(
                            &result->finish,
                            result->module,
                            "ormi_finish"
                        ),
                        "cuModuleGetFunction"
                    );
                }

                int minimum_blocks = 0;
                check_driver(
                    cuOccupancyMaxPotentialBlockSize(
                        &minimum_blocks,
                        &result->threads,
                        result->function,
                        nullptr,
                        0,
                        0
                    ),
                    "cuOccupancyMaxPotentialBlockSize"
                );

                if (result->tiled) {
                    result->threads =
                        static_cast<int>(result->tiled->width);
                } else {
                    int power_of_two = 1;
                    while (power_of_two <= result->threads / 2) {
                        power_of_two *= 2;
                    }
                    result->threads = power_of_two;
                }

                int active_blocks = 0;
                const auto shared_bytes =
                    expression.reduction && !result->tiled
                    ? static_cast<std::size_t>(result->threads) *
                        (expression.dtype == core::DType::float64
                            ? sizeof(double)
                            : sizeof(float))
                    : std::size_t{0};
                check_driver(
                    cuOccupancyMaxActiveBlocksPerMultiprocessor(
                        &active_blocks,
                        result->function,
                        result->threads,
                        shared_bytes
                    ),
                    "cuOccupancyMaxActiveBlocksPerMultiprocessor"
                );

                result->max_blocks = std::max(
                    1,
                    cuda_state.multiprocessors() * active_blocks
                );
            } catch (...) {
                (void)cuModuleUnload(result->module);
                result->module = nullptr;
                throw;
            }

            // Published modules remain valid until the CUDA context is torn down.
            return result;
        }
    }

    std::shared_ptr<Compiled> compile(
        const core::runtime::fusion::Expr& expression,
        const State& cuda_state
    ) {
        using Cache = std::unordered_map<std::uint64_t, std::vector<Entry>>;
        static std::mutex mutex;
        static std::atomic<std::shared_ptr<const Cache>> published{
            std::make_shared<const Cache>()
        };

        const auto device = cuda_state.index();
        const auto major = cuda_state.compute_major();
        const auto minor = cuda_state.compute_minor();
        auto hash = core::runtime::fusion::Key::hash(expression);
        hash = combine(hash, static_cast<std::uint64_t>(device));
        hash = combine(hash, static_cast<std::uint64_t>(major));
        hash = combine(hash, static_cast<std::uint64_t>(minor));

        auto current = published.load(std::memory_order_acquire);
        if (const auto found = current->find(hash); found != current->end()) {
            for (const auto& entry : found->second) {
                if (entry.matches(device, major, minor, expression)) {
                    return entry.compiled;
                }
            }
        }

        std::lock_guard lock(mutex);
        current = published.load(std::memory_order_acquire);
        if (const auto found = current->find(hash); found != current->end()) {
            for (const auto& entry : found->second) {
                if (entry.matches(device, major, minor, expression)) {
                    return entry.compiled;
                }
            }
        }

        auto compiled = build(expression, cuda_state, major, minor);
        auto next = std::make_shared<Cache>(*current);
        (*next)[hash].push_back(Entry{
            device,
            major,
            minor,
            expression,
            compiled
        });
        published.store(std::move(next), std::memory_order_release);
        return compiled;
    }
}
