#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <ormi/core/error.h>

#include "backends/cpu/jit/artifact.h"
#include "backends/cpu/jit/cache.h"
#include "backends/cpu/jit/compiler.h"
#include "backends/cpu/jit/math.h"
#include "backends/cpu/jit/target.h"
#include "platform/platform.h"

namespace ormi::backends::cpu::jit::artifact {
    namespace {
        void build_source(
            const std::filesystem::path& source,
            const std::filesystem::path& output
        ) {
            const auto& compiler = toolchain();
            build(
                compiler,
                selected_target(),
                selected_math(compiler.id),
                source,
                output
            );
        }

        std::string symbol(std::size_t index, std::size_t count) {
            return count == 1
                ? "ormi_kernel"
                : "ormi_kernel_" + std::to_string(index);
        }

        std::vector<std::shared_ptr<Compiled>> load(
            const std::filesystem::path& path,
            std::span<const Kernel> kernels,
            Math functions
        ) {
            auto library = std::make_shared<platform::Library>(path);
            std::vector<std::shared_ptr<Compiled>> result;
            result.reserve(kernels.size());
            for (std::size_t i = 0; i < kernels.size(); ++i) {
                const auto kernel = kernels[i];
                auto function = reinterpret_cast<Function>(
                    library->symbol(symbol(i, kernels.size()))
                );
                result.push_back(std::make_shared<Compiled>(Compiled{
                    library,
                    function,
                    functions.functions,
                    kernel.math && functions.functions
                        ? kernel.math_work
                        : kernel.work,
                    kernel.unit,
                    kernel.range,
                    kernel.partial
                }));
            }
            return result;
        }

        std::vector<std::shared_ptr<Compiled>> cached(
            std::string_view source,
            std::span<const Kernel> kernels,
            Math functions
        ) {
            const auto identity = context(functions);
            auto value = cache::find(source, identity);
            if (!value) {
                return {};
            }
            try {
                return load(value->library, kernels, functions);
            } catch (const jitError&) {
                value.reset();
                cache::discard(source, identity);
                return {};
            }
        }

        std::vector<std::shared_ptr<Compiled>> find(
            std::string_view source,
            std::span<const Kernel> kernels
        ) {
            const auto direct = selected_target().avx2
                ? math(MathABI::direct)
                : Math{};
            if (direct.functions) {
                if (auto result = cached(source, kernels, direct); !result.empty()) {
                    return result;
                }
            }

            const auto buffered = selected_target().avx2
                ? math(MathABI::buffered)
                : Math{};
            if (buffered.functions) {
                if (auto result = cached(source, kernels, buffered); !result.empty()) {
                    return result;
                }
            }

            return cached(source, kernels, Math{});
        }
    }

    std::shared_ptr<Compiled> find(
        std::string_view source,
        Kernel kernel
    ) {
        const std::span kernels(&kernel, 1);
        auto result = find(source, kernels);
        return result.empty() ? nullptr : std::move(result.front());
    }

    std::vector<std::shared_ptr<Compiled>> get(
        std::string_view source,
        std::span<const Kernel> kernels
    ) {
        if (kernels.empty()) {
            ORMI_THROW(OperationError, "Cannot load an empty CPU JIT artifact.");
        }
        if (auto result = find(source, kernels); !result.empty()) {
            return result;
        }

        const auto functions = selected_math(toolchain().id);
        const auto identity = context(functions);
        auto value = cache::get(source, identity, build_source);
        try {
            return load(value.library, kernels, functions);
        } catch (const jitError&) {
            if (!value.cached) {
                throw;
            }
        }

        cache::discard(source, identity);
        auto repaired = cache::get(source, identity, build_source);
        return load(repaired.library, kernels, functions);
    }
}
