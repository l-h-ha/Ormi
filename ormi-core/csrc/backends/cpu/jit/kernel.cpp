#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <ormi/core/error.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/jit/cache.h"
#include "backends/cpu/jit/codegen.h"
#include "backends/cpu/jit/compiler.h"
#include "backends/cpu/jit/kernel.h"
#include "platform/platform.h"

namespace ormi::backends::cpu::jit {
    namespace {
        using Function = void (*)(void*, const void* const*, unsigned long long);

        struct Compiled {
            platform::Library library;
            Function function;
        };

        std::uint64_t combine(std::uint64_t hash, std::uint64_t value) {
            hash ^= value;
            hash *= 1099511628211ull;
            return hash;
        }

        std::uint64_t structural_hash(const core::runtime::fusion::Expr& expression) {
            auto result = combine(
                14695981039346656037ull,
                static_cast<std::uint64_t>(expression.dtype)
            );
            result = combine(result, expression.output);
            result = combine(result, expression.nodes.size());

            for (const auto& node : expression.nodes) {
                result = combine(result, node.index());
                std::visit(
                    [&result](const auto& value) {
                        using T = std::decay_t<decltype(value)>;
                        if constexpr (std::is_same_v<T, core::runtime::fusion::Input>) {
                            result = combine(result, value.index);
                        } else if constexpr (std::is_same_v<T, core::runtime::fusion::Constant>) {
                            result = combine(result, static_cast<std::uint64_t>(value));
                        } else if constexpr (std::is_same_v<T, core::runtime::fusion::Unary>) {
                            result = combine(result, static_cast<std::uint64_t>(value.op));
                            result = combine(result, value.input);
                        } else {
                            result = combine(result, static_cast<std::uint64_t>(value.op));
                            result = combine(result, value.lhs);
                            result = combine(result, value.rhs);
                        }
                    },
                    node
                );
            }

            return result;
        }

        std::vector<std::string> command(
            const Toolchain& compiler,
            const std::filesystem::path& source,
            const std::filesystem::path& output
        ) {
            if (compiler.id == CXX::msvc || compiler.id == CXX::clang_cl) {
                auto object = source;
                object.replace_extension(".obj");
                return {
                    "/nologo",
                    "/std:c++20",
                    "/O2",
                    "/LD",
                    "/EHsc",
                    "/Fo:" + object.string(),
                    source.string(),
                    "/Fe:" + output.string(),
                    "/link",
                    "/NOIMPLIB"
                };
            }

            std::vector<std::string> arguments{
                "-std=c++20",
                "-O3",
                "-DNDEBUG"
            };
            const auto shared = platform::shared_flags();
            arguments.insert(arguments.end(), shared.begin(), shared.end());
            arguments.push_back("-o");
            arguments.push_back(output.string());
            arguments.push_back(source.string());
            return arguments;
        }

        void build(
            const std::filesystem::path& source,
            const std::filesystem::path& output
        ) {
            static const Toolchain compiler = resolve();
            const auto result = run(compiler, command(compiler, source, output));
            auto object = source;
            object.replace_extension(".obj");
            std::error_code ignored;
            std::filesystem::remove(object, ignored);

            if (result.code != 0 || !std::filesystem::is_regular_file(output)) {
                ORMI_THROW(
                    jitError,
                    "CPU JIT compiler failed with exit code " + std::to_string(result.code) +
                    ".\nCompiler: " + compiler.executable.string() +
                    "\n" + result.output
                );
            }
        }

        std::shared_ptr<Compiled> load(const std::filesystem::path& path) {
            platform::Library library(path);
            auto function = reinterpret_cast<Function>(library.symbol("ormi_kernel"));
            return std::make_shared<Compiled>(Compiled{std::move(library), function});
        }

        std::shared_ptr<Compiled> compile(std::string_view generated) {
            bool cached = false;

            {
                auto artifact = cache::get(generated, build);
                cached = artifact.cached;

                try {
                    return load(artifact.library);
                } catch (const jitError&) {
                    if (!cached) {
                        throw;
                    }
                }
            }

            cache::discard(generated);
            auto repaired = cache::get(generated, build);
            return load(repaired.library);
        }

        struct Entry {
            Entry(
                const core::runtime::fusion::Expr& expression,
                std::shared_ptr<Compiled> compiled_kernel
            )
                : dtype(expression.dtype),
                  nodes(expression.nodes),
                  output(expression.output),
                  compiled(std::move(compiled_kernel)) {}

            bool matches(const core::runtime::fusion::Expr& expression) const {
                return
                    dtype == expression.dtype &&
                    output == expression.output &&
                    nodes == expression.nodes;
            }

            core::DType dtype;
            std::vector<core::runtime::fusion::Node> nodes;
            core::runtime::fusion::Index output;
            std::shared_ptr<Compiled> compiled;
        };

        std::shared_ptr<Compiled> kernel(const core::runtime::fusion::Expr& expression) {
            static std::mutex mutex;
            static std::unordered_map<std::uint64_t, std::vector<Entry>> entries;
            const auto hash = structural_hash(expression);
            std::lock_guard lock(mutex);
            auto& candidates = entries[hash];

            for (const auto& entry : candidates) {
                if (entry.matches(expression)) {
                    return entry.compiled;
                }
            }

            auto compiled = compile(emit(expression));
            candidates.emplace_back(expression, compiled);
            return compiled;
        }
    }

    void execute(core::Tensor& out, const core::runtime::fusion::Expr& expression) {
        auto compiled = kernel(expression);
        std::vector<const void*> inputs;
        inputs.reserve(expression.inputs.size());

        for (const auto* input : expression.inputs) {
            inputs.push_back(input->data());
        }

        compiled->function(
            out.data(),
            inputs.data(),
            static_cast<unsigned long long>(out.numel())
        );
    }
}
