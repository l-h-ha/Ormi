#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/error.h>

#include "backends/cpu/jit/compiler.h"
#include "backends/cpu/jit/target.h"
#include "platform/platform.h"

namespace {
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

    class EnvironmentGuard {
    public:
        explicit EnvironmentGuard(const char* name) : name_(name) {
            if (const char* value = std::getenv(name)) {
                value_ = value;
            }
        }

        ~EnvironmentGuard() {
            environment(name_, value_ ? value_->c_str() : "");
        }

        EnvironmentGuard(const EnvironmentGuard&) = delete;
        EnvironmentGuard& operator=(const EnvironmentGuard&) = delete;

    private:
        const char* name_;
        std::optional<std::string> value_;
    };

    TEST(CPUJITTargetTest, selectsBestAvailableTargetAutomatically) {
        const EnvironmentGuard target("ORMI_CPU_JIT_TARGET");
        environment("ORMI_CPU_JIT_TARGET", "");

        const ormi::backends::cpu::jit::Target expected{
            .avx2 = ormi::platform::supports_avx2(),
            .f16c = ormi::platform::supports_avx2() &&
                ormi::platform::supports_f16c()
        };

        EXPECT_EQ(ormi::backends::cpu::jit::select_target(), expected);
    }

    TEST(CPUJITTargetTest, acceptsCaseInsensitiveBaselineOverride) {
        const EnvironmentGuard target("ORMI_CPU_JIT_TARGET");
        environment("ORMI_CPU_JIT_TARGET", "BaSeLiNe");

        EXPECT_EQ(
            ormi::backends::cpu::jit::select_target(),
            ormi::backends::cpu::jit::Target{}
        );
    }

    TEST(CPUJITTargetTest, validatesExplicitAVX2Override) {
        const EnvironmentGuard target("ORMI_CPU_JIT_TARGET");
        environment("ORMI_CPU_JIT_TARGET", "AVX2");

        if (ormi::platform::supports_avx2()) {
            EXPECT_EQ(
                ormi::backends::cpu::jit::select_target(),
                (ormi::backends::cpu::jit::Target{
                    .avx2 = true,
                    .f16c = ormi::platform::supports_f16c()
                })
            );
        } else {
            EXPECT_THROW(
                (void)ormi::backends::cpu::jit::select_target(),
                ormi::jitError
            );
        }
    }

    TEST(CPUJITTargetTest, rejectsUnknownOverride) {
        const EnvironmentGuard target("ORMI_CPU_JIT_TARGET");
        environment("ORMI_CPU_JIT_TARGET", "fastest");

        EXPECT_THROW(
            (void)ormi::backends::cpu::jit::select_target(),
            ormi::jitError
        );
    }

    TEST(CPUJITTargetTest, mapsTargetToCompilerSyntax) {
        using ormi::backends::cpu::jit::CXX;
        using ormi::backends::cpu::jit::Target;
        using ormi::backends::cpu::jit::target_flags;

        const Target avx2{.avx2 = true};
        const Target f16c{.avx2 = true, .f16c = true};

        EXPECT_TRUE(target_flags(CXX::gcc, {}).empty());
        EXPECT_EQ(
            target_flags(CXX::gcc, avx2),
            (std::vector<std::string>{"-mavx2", "-DORMI_USE_AVX2=1"})
        );
        EXPECT_EQ(
            target_flags(CXX::gcc, f16c),
            (std::vector<std::string>{
                "-mavx2",
                "-DORMI_USE_AVX2=1",
                "-mf16c",
                "-DORMI_USE_F16C=1"
            })
        );
        EXPECT_EQ(
            target_flags(CXX::clang, avx2),
            (std::vector<std::string>{"-mavx2", "-DORMI_USE_AVX2=1"})
        );
        EXPECT_EQ(
            target_flags(CXX::clang_cl, f16c),
            (std::vector<std::string>{
                "/arch:AVX2",
                "/DORMI_USE_AVX2=1",
                "/DORMI_USE_F16C=1"
            })
        );
        EXPECT_EQ(
            target_flags(CXX::msvc, avx2),
            (std::vector<std::string>{
                "/arch:AVX2",
                "/DORMI_USE_AVX2=1"
            })
        );
        EXPECT_THROW(
            (void)target_flags(CXX::gcc, Target{.f16c = true}),
            ormi::jitError
        );
    }
}
