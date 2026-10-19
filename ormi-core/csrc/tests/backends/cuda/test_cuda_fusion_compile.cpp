#include <cstdlib>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <nvrtc.h>

#include <ormi/core/error.h>
#include <ormi/core/memory/ops.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/tensor.h>

#include "backends/cuda/fusion/cache.h"
#include "backends/cuda/fusion/codegen.h"
#include "backends/cuda/fusion/nvrtc.h"
#include "backends/cuda/fusion/tile.h"
#include "core/runtime/device.h"
#include "core/runtime/fusion/expr.h"
#include "platform/platform.h"

namespace {
    using namespace ormi::core;
    namespace fusion = ormi::backends::cuda::fusion;

    void environment(const char* name, const char* value) {
#if defined(_WIN32)
        _putenv_s(name, value);
#else
        if (*value) {
            setenv(name, value, 1);
        } else {
            unsetenv(name);
        }
#endif
    }

    std::optional<std::string> environment(const char* name) {
        const auto* value = std::getenv(name);
        return value ? std::optional<std::string>(value) : std::nullopt;
    }

    class CUDAFusionCompileTest : public ::testing::Test {
    protected:
        void SetUp() override {
            int devices = 0;
            const auto status = cudaGetDeviceCount(&devices);
            if (status != cudaSuccess) {
                GTEST_SKIP() << cudaGetErrorString(status);
            }
            if (devices == 0) {
                GTEST_SKIP() << "No CUDA device is available.";
            }
            ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
            ASSERT_EQ(cudaDeviceGetAttribute(&major_, cudaDevAttrComputeCapabilityMajor, 0), cudaSuccess);
            ASSERT_EQ(cudaDeviceGetAttribute(&minor_, cudaDevAttrComputeCapabilityMinor, 0), cudaSuccess);
            int major = 0;
            int minor = 0;
            ASSERT_EQ(nvrtcVersion(&major, &minor), NVRTC_SUCCESS);
            context_ = "compute_" + std::to_string(major_) + std::to_string(minor_) +
                "\nnvrtc_" + std::to_string(major) + "." + std::to_string(minor);

            directory_ = std::filesystem::temp_directory_path() /
                ("ormi-cuda-fusion-compile-test-" +
                    std::to_string(ormi::platform::process_id()) + "-" +
                    ::testing::UnitTest::GetInstance()->current_test_info()->name());
            previous_directory_ = environment("ORMI_CUDA_JIT_CACHE_DIR");
            previous_limit_ = environment("ORMI_CUDA_JIT_CACHE_MAX_BYTES");
            environment("ORMI_CUDA_JIT_CACHE_DIR", directory_.string().c_str());
            environment("ORMI_CUDA_JIT_CACHE_MAX_BYTES", "");
        }

        void TearDown() override {
            if (directory_.empty()) {
                return;
            }
            std::error_code error;
            std::filesystem::remove_all(directory_, error);
            EXPECT_FALSE(error) << error.message();
            environment("ORMI_CUDA_JIT_CACHE_DIR",
                previous_directory_ ? previous_directory_->c_str() : "");
            environment("ORMI_CUDA_JIT_CACHE_MAX_BYTES",
                previous_limit_ ? previous_limit_->c_str() : "");
        }

        std::string ptx(const char* source) const {
            nvrtcProgram program = nullptr;
            if (nvrtcCreateProgram(&program, source, "failure.cu", 0, nullptr, nullptr) != NVRTC_SUCCESS) {
                throw std::runtime_error("Cannot create test NVRTC program.");
            }
            std::string result;
            try {
                const auto architecture = "--gpu-architecture=compute_" +
                    std::to_string(major_) + std::to_string(minor_);
                const char* options[]{"--std=c++17", architecture.c_str()};
                if (nvrtcCompileProgram(program, 2, options) != NVRTC_SUCCESS) {
                    throw std::runtime_error("Cannot compile test PTX.");
                }
                std::size_t size = 0;
                if (nvrtcGetPTXSize(program, &size) != NVRTC_SUCCESS) {
                    throw std::runtime_error("Cannot size test PTX.");
                }
                result.resize(size);
                if (nvrtcGetPTX(program, result.data()) != NVRTC_SUCCESS) {
                    throw std::runtime_error("Cannot retrieve test PTX.");
                }
            } catch (...) {
                (void)nvrtcDestroyProgram(&program);
                throw;
            }
            (void)nvrtcDestroyProgram(&program);
            return result;
        }

        void reject_and_retry(Shape shape, bool reduce, const std::string& bad_ptx, const char* operation) {
            const auto device = Device::from("cuda");
            const auto input = ops::full(std::move(shape), 0.5f, device);
            input->realize();
            auto graph = ops::add(ops::mul(input, input), input);
            if (reduce) {
                graph = ops::sum(graph, {0});
            }
            const auto expression = runtime::fusion::build(*graph);
            ASSERT_TRUE(expression);
            const auto output = ops::full(graph->shape(), 0.0f, device);
            output->realize();
            const auto source = fusion::emit(*expression);
            const auto artifact = fusion::cache::get(source, context_, [&] { return bad_ptx; });
            ASSERT_FALSE(artifact.cached);
            ASSERT_EQ(artifact.ptx, bad_ptx);

            // A failed module must never become an in-memory cache hit.
            for (int attempt = 0; attempt < 3; ++attempt) {
                try {
                    auto setup = runtime::backend(device).fusion.prepare(*output, {}, *expression);
                    FAIL() << "Invalid fusion artifact was accepted.";
                } catch (const ormi::BackendError& error) {
                    EXPECT_NE(std::string(error.what()).find(operation), std::string::npos)
                        << error.what();
                }
                EXPECT_FALSE(graph->realized());
            }
            ASSERT_EQ(fusion::cache::info().entries, 1u);
            fusion::cache::clear();
            ASSERT_EQ(fusion::cache::info().entries, 0u);

            auto setup = runtime::backend(device).fusion.prepare(*output, {}, *expression);
            setup.run(*output);
            std::vector<float> values(output->numel());
            memory::copy(values.data(), Device::cpu(), output->data(), device, output->bytes());
            const auto expected = reduce
                ? 0.75f * static_cast<float>(input->shape()[0])
                : 0.75f;
            for (const auto value : values) {
                EXPECT_FLOAT_EQ(value, expected);
            }
            EXPECT_FALSE(graph->realized());
            ASSERT_EQ(fusion::cache::info().entries, 1u);

            // Reuse the published kernel even when its disk artifact is absent.
            fusion::cache::clear();
            auto cached = runtime::backend(device).fusion.prepare(*output, {}, *expression);
            cached.run(*output);
            memory::copy(values.data(), Device::cpu(), output->data(), device, output->bytes());
            for (const auto value : values) {
                EXPECT_FLOAT_EQ(value, expected);
            }
            EXPECT_EQ(fusion::cache::info().entries, 0u);
        }

        int major_ = 0;
        int minor_ = 0;
        std::string context_;
        std::filesystem::path directory_;
        std::optional<std::string> previous_directory_;
        std::optional<std::string> previous_limit_;
    };

    TEST_F(CUDAFusionCompileTest, rejectsInvalidPTXAndRetries) {
        reject_and_retry({97}, false, "invalid PTX", "cuModuleLoadData");
    }

    TEST_F(CUDAFusionCompileTest, failedCompilationDoesNotPublishAndCanRetry) {
        const std::string source = "extern \"C\" __global__ void ormi_kernel() {}";
        for (int attempt = 0; attempt < 3; ++attempt) {
            EXPECT_THROW((void)fusion::cache::get(source, context_, [&] {
                return fusion::nvrtc::compile("#error ORMI_CACHE_BUILD_FAILURE\n", major_, minor_);
            }), ormi::jitError);
            EXPECT_EQ(fusion::cache::info().entries, 0u);
        }

        const auto artifact = fusion::cache::get(source, context_, [&] {
            return fusion::nvrtc::compile(source, major_, minor_);
        });
        EXPECT_FALSE(artifact.cached);
        EXPECT_NE(artifact.ptx.find(".entry ormi_kernel"), std::string::npos);
        const auto cached = fusion::cache::get(source, context_, [] {
            ADD_FAILURE() << "A valid NVRTC artifact was rebuilt.";
            return std::string{};
        });
        EXPECT_TRUE(cached.cached);
        EXPECT_EQ(cached.ptx, artifact.ptx);
        EXPECT_EQ(fusion::cache::info().entries, 1u);
    }

    TEST_F(CUDAFusionCompileTest, rejectsMissingKernelAndRetries) {
        reject_and_retry({101}, false,
            ptx("extern \"C\" __global__ void unrelated() {}"), "cuModuleGetFunction");
    }

    TEST_F(CUDAFusionCompileTest, rejectsMissingFinishAndRetries) {
        reject_and_retry({103}, true,
            ptx("extern \"C\" __global__ void ormi_kernel() {}"), "cuModuleGetFunction");
    }

    TEST_F(CUDAFusionCompileTest, loadsTiledKernelWithoutFinish) {
        const auto device = Device::from("cuda");
        const auto input = ops::full({5, 257}, 0.5f, device);
        input->realize();
        const auto graph = ops::sum(ops::add(ops::mul(input, input), input), {0});
        const auto expression = runtime::fusion::build(*graph);
        ASSERT_TRUE(expression);
        ASSERT_TRUE(fusion::tile(*expression));
        EXPECT_EQ(fusion::emit(*expression).find("ormi_finish"), std::string::npos);

        const auto output = ops::full(graph->shape(), 0.0f, device);
        output->realize();
        auto setup = runtime::backend(device).fusion.prepare(*output, {}, *expression);
        setup.run(*output);
        std::vector<float> values(output->numel());
        memory::copy(values.data(), Device::cpu(), output->data(), device, output->bytes());
        for (const auto value : values) {
            EXPECT_FLOAT_EQ(value, 3.75f);
        }
    }
}
