#include <array>
#include <exception>
#include <string>
#include <thread>
#include <vector>
#include <gtest/gtest.h>

#include <ormi/core/error.h>
#include <ormi/backend/api.h>
#include <ormi/backend/id.h>

#include "core/runtime/backends.h"

namespace {
    TEST(BackendLoadingTest, resolvesOptionalCUDAOnce) {
        auto& backends = ormi::core::runtime::Backends::instance();
        const auto cuda = backends.find("cuda");

        ASSERT_NE(cuda, ormi::core::runtime::backend_invalid);
        EXPECT_FALSE(backends.available(cuda));

    #ifdef ORMI_ENABLE_CUDA
        std::array<const ormi::core::runtime::Backend*, 8> loaded{};
        std::array<std::exception_ptr, 8> errors{};
        std::vector<std::thread> threads;
        threads.reserve(loaded.size());

        for (std::size_t i = 0; i < loaded.size(); ++i) {
            threads.emplace_back([&backends, &loaded, &errors, cuda, i] {
                try {
                    loaded[i] = &backends.require(cuda);
                } catch (...) {
                    errors[i] = std::current_exception();
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        EXPECT_TRUE(backends.available(cuda));
        ASSERT_NE(loaded[0], nullptr);
        for (std::size_t i = 0; i < loaded.size(); ++i) {
            EXPECT_EQ(errors[i], nullptr);
            EXPECT_EQ(loaded[i], loaded[0]);
        }
        EXPECT_EQ(loaded[0]->abi, ormi::core::runtime::backend_abi);
    #else
        std::string first_error;
        try {
            (void)backends.require(cuda);
            FAIL() << "Expected DeviceError.";
        } catch (const ormi::DeviceError& error) {
            first_error = error.what();
        }

        EXPECT_FALSE(first_error.empty());
        EXPECT_NE(first_error.find("Cannot load backend 'cuda'"), std::string::npos);
        EXPECT_FALSE(backends.available(cuda));
        EXPECT_THROW(
            (void)backends.require(cuda),
            ormi::DeviceError
        );
    #endif
    }
}
