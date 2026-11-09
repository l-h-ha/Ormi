#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include <cuda.h>
#include <gtest/gtest.h>

#include <ormi/core/error.h>

#include "backends/cuda/fusion/module.h"
#include "backends/cuda/state.h"

namespace {
    namespace core = ormi::core;
    namespace fusion = ormi::backends::cuda::fusion;

    enum class Step { load, kernel, finish, potential, active, unload };

    // Only this test target replaces driver calls and device state.
    struct Driver {
        std::optional<Step> failure;
        std::vector<Step> calls;
        const void* image = nullptr;
        int suggested_threads = 384;
        int active_blocks = 3;
        int requested_threads = 0;
        std::size_t shared_bytes = 0;
        CUresult unload_status = CUDA_SUCCESS;
    } driver;

    int module_token;
    int kernel_token;
    int finish_token;

    class CUDAFusionModuleTest : public testing::Test {
    protected:
        void SetUp() override {
            driver = {};
        }

        core::runtime::fusion::Expr expression(core::DType dtype, bool reduce) {
            core::runtime::fusion::Expr result{};
            result.dtype = dtype;
            if (reduce) {
                result.reduction = core::runtime::fusion::Reduction{
                    core::ops::ReduceOp::sum, {1, {}}, {4097, {}}
                };
            }
            return result;
        }

        void reject_and_retry(Step failure, const char* operation) {
            const ormi::backends::cuda::State state(0);
            const auto expr = expression(core::DType::float32, true);
            const int unloads = failure == Step::load ? 0 : 1;
            driver.failure = failure;
            try {
                (void)fusion::load("test PTX", expr, state, std::nullopt);
                FAIL() << "A driver failure was accepted.";
            } catch (const ormi::BackendError& error) {
                EXPECT_NE(std::string(error.what()).find(operation), std::string::npos);
            }
            EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(), Step::unload), unloads);
            EXPECT_EQ(driver.calls.back(), failure == Step::load ? failure : Step::unload);

            driver.failure.reset();
            auto result = fusion::load("test PTX", expr, state, std::nullopt);
            ASSERT_NE(result->module, nullptr);
            EXPECT_NE(result->function, nullptr);
            EXPECT_NE(result->finish, nullptr);
            EXPECT_EQ(result->threads, 256);
            EXPECT_EQ(result->max_blocks, 24);
            result.reset();
            EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(), Step::unload), unloads);
        }
    };

    TEST_F(CUDAFusionModuleTest, failedModuleLoadCanRetryWithoutUnloading) {
        reject_and_retry(Step::load, "cuModuleLoadData");
    }

    TEST_F(CUDAFusionModuleTest, missingKernelUnloadsAndCanRetry) {
        reject_and_retry(Step::kernel, "cuModuleGetFunction");
    }

    TEST_F(CUDAFusionModuleTest, missingFinishUnloadsAndCanRetry) {
        reject_and_retry(Step::finish, "cuModuleGetFunction");
    }

    TEST_F(CUDAFusionModuleTest, failedPotentialOccupancyUnloadsAndCanRetry) {
        reject_and_retry(Step::potential, "cuOccupancyMaxPotentialBlockSize");
    }

    TEST_F(CUDAFusionModuleTest, failedActiveOccupancyUnloadsAndCanRetry) {
        reject_and_retry(Step::active, "cuOccupancyMaxActiveBlocksPerMultiprocessor");
    }

    TEST_F(CUDAFusionModuleTest, unloadFailurePreservesOriginalErrorAndCanRetry) {
        driver.unload_status = CUDA_ERROR_INVALID_CONTEXT;
        reject_and_retry(Step::potential, "cuOccupancyMaxPotentialBlockSize");
    }

    TEST_F(CUDAFusionModuleTest, pointwiseUsesNoFinishOrSharedMemory) {
        const ormi::backends::cuda::State state(0);
        const auto expr = expression(core::DType::float32, false);
        const std::string source = "test PTX";
        auto result = fusion::load(source, expr, state, std::nullopt);
        EXPECT_EQ(driver.image, source.data());
        EXPECT_EQ(result->module, reinterpret_cast<CUmodule>(&module_token));
        EXPECT_EQ(result->function, reinterpret_cast<CUfunction>(&kernel_token));
        EXPECT_EQ(result->finish, nullptr);
        EXPECT_FALSE(result->tiled);
        EXPECT_EQ(result->threads, 256);
        EXPECT_EQ(result->max_blocks, 24);
        EXPECT_EQ(driver.requested_threads, 256);
        EXPECT_EQ(driver.shared_bytes, 0u);
        const std::vector expected{Step::load, Step::kernel, Step::potential, Step::active};
        EXPECT_EQ(driver.calls, expected);
        result.reset();
        EXPECT_EQ(driver.calls, expected);
    }

    TEST_F(CUDAFusionModuleTest, reductionsUseAccumulatorWidthForEveryFloatingDType) {
        const ormi::backends::cuda::State state(0);
        for (const auto dtype : {core::DType::float16, core::DType::float32, core::DType::float64}) {
            driver = {};
            const auto expr = expression(dtype, true);
            const auto result = fusion::load("test PTX", expr, state, std::nullopt);
            EXPECT_EQ(result->finish, reinterpret_cast<CUfunction>(&finish_token));
            EXPECT_EQ(driver.requested_threads, 256);
            EXPECT_EQ(driver.shared_bytes, 256 * (dtype == core::DType::float64 ? sizeof(double) : sizeof(float)));
            const std::vector expected{Step::load, Step::kernel, Step::finish, Step::potential, Step::active};
            EXPECT_EQ(driver.calls, expected);
        }
    }

    TEST_F(CUDAFusionModuleTest, tiledReductionPreservesMetadataAndOverridesThreadCount) {
        const ormi::backends::cuda::State state(0);
        const auto expr = expression(core::DType::float16, true);
        const fusion::Tile tiled{0, 64, 9, {1}, {3}, {{0, 1}}};
        const auto result = fusion::load("test PTX", expr, state, tiled);
        ASSERT_TRUE(result->tiled);
        EXPECT_EQ(result->tiled->axis, tiled.axis);
        EXPECT_EQ(result->tiled->width, tiled.width);
        EXPECT_EQ(result->tiled->tasks, tiled.tasks);
        EXPECT_EQ(result->tiled->output, tiled.output);
        EXPECT_EQ(result->tiled->reduced, tiled.reduced);
        EXPECT_EQ(result->tiled->inputs, tiled.inputs);
        EXPECT_EQ(result->threads, 64);
        EXPECT_EQ(driver.requested_threads, 64);
        EXPECT_EQ(driver.shared_bytes, 0u);
        EXPECT_EQ(result->finish, nullptr);
        const std::vector expected{Step::load, Step::kernel, Step::potential, Step::active};
        EXPECT_EQ(driver.calls, expected);
    }

    TEST_F(CUDAFusionModuleTest, blockLimitHasOneBlockMinimum) {
        const ormi::backends::cuda::State state(0);
        driver.active_blocks = 0;
        const auto result = fusion::load("test PTX", expression(core::DType::float64, false), state, std::nullopt);
        EXPECT_EQ(result->max_blocks, 1);
    }

    TEST_F(CUDAFusionModuleTest, keepsOneThreadWhenOccupancySuggestsNoThreads) {
        const ormi::backends::cuda::State state(0);
        driver.suggested_threads = 0;
        const auto result = fusion::load("test PTX", expression(core::DType::float32, false), state, std::nullopt);
        EXPECT_EQ(result->threads, 1);
        EXPECT_EQ(driver.requested_threads, 1);
    }
}

namespace ormi::backends::cuda {
    State::State(int index) : index_(index) {}
    State::~State() noexcept = default;

    int State::multiprocessors() const noexcept {
        return 8;
    }
}

extern "C" CUresult CUDAAPI cuModuleLoadData(CUmodule* module, const void* image) {
    driver.calls.push_back(Step::load);
    driver.image = image;
    if (driver.failure == Step::load) {
        return CUDA_ERROR_INVALID_PTX;
    }
    *module = reinterpret_cast<CUmodule>(&module_token);
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuModuleGetFunction(CUfunction* function, CUmodule module, const char* name) {
    EXPECT_EQ(module, reinterpret_cast<CUmodule>(&module_token));
    const auto step = std::string(name) == "ormi_kernel" ? Step::kernel : Step::finish;
    EXPECT_TRUE(std::string(name) == "ormi_kernel" || std::string(name) == "ormi_finish");
    driver.calls.push_back(step);
    if (driver.failure == step) {
        return CUDA_ERROR_NOT_FOUND;
    }
    *function = reinterpret_cast<CUfunction>(step == Step::kernel ? &kernel_token : &finish_token);
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuModuleUnload(CUmodule module) {
    EXPECT_EQ(module, reinterpret_cast<CUmodule>(&module_token));
    driver.calls.push_back(Step::unload);
    return driver.unload_status;
}

extern "C" CUresult CUDAAPI cuOccupancyMaxPotentialBlockSize(
    int* minimum_blocks, int* threads, CUfunction function,
    CUoccupancyB2DSize callback, std::size_t shared_bytes, int limit
) {
    EXPECT_EQ(function, reinterpret_cast<CUfunction>(&kernel_token));
    EXPECT_EQ(callback, nullptr);
    EXPECT_EQ(shared_bytes, 0u);
    EXPECT_EQ(limit, 0);
    driver.calls.push_back(Step::potential);
    if (driver.failure == Step::potential) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *minimum_blocks = 1;
    *threads = driver.suggested_threads;
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessor(
    int* blocks, CUfunction function, int threads, std::size_t shared_bytes
) {
    EXPECT_EQ(function, reinterpret_cast<CUfunction>(&kernel_token));
    driver.calls.push_back(Step::active);
    driver.requested_threads = threads;
    driver.shared_bytes = shared_bytes;
    if (driver.failure == Step::active) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *blocks = driver.active_blocks;
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuGetErrorName(CUresult, const char** name) {
    *name = "CUDA_ERROR_TEST_FAILURE";
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuGetErrorString(CUresult, const char** description) {
    *description = "Controlled module test failure.";
    return CUDA_SUCCESS;
}
