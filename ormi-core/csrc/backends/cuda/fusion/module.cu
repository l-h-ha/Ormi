#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <cuda.h>

#include "backends/cuda/check.cuh"
#include "backends/cuda/fusion/module.h"
#include "backends/cuda/state.h"

namespace ormi::backends::cuda::fusion {
    std::shared_ptr<Compiled> load(
        const std::string& ptx,
        const core::runtime::fusion::Expr& expression,
        const State& cuda_state,
        std::optional<Tile> tiled
    ) {
        auto result = std::make_shared<Compiled>();
        result->tiled = std::move(tiled);

        check_driver(
            cuModuleLoadData(&result->module, ptx.data()),
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
