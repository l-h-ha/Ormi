#include <stdexcept>
#include <cmath>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <filesystem>
#include <mutex>

#include "jit.h"
#include "compiler.h"
#include "core/buffer.h"
#include "core/shape.h"
#include "lowering_registry.h"
#include "core/error.h"

#ifdef _WIN32
    #include <windows.h>
    const std::string OS_LIB_EXT = ".dll";

    void* load_shared_library(const std::string& path) {
        return (void*)LoadLibraryA(path.c_str());
    }

    void* get_symbol(void* handle, const std::string& symbol) {
        return (void*)GetProcAddress((HMODULE)handle, symbol.c_str());
    }

    std::string get_dl_error() {
        return "Win32 Error Code: " + std::to_string(GetLastError());
    }
#else
    #include <dlfcn.h>
    const std::string OS_LIB_EXT = ".so";

    void* load_shared_library(const std::string& path) {
        return dlopen(path.c_str(), RTLD_NOW);
    }

    void* get_symbol(void* handle, const std::string& symbol) {
        return dlsym(handle, symbol.c_str());
    }

    std::string get_dl_error() {
        const char* err = dlerror();
        return err ? std::string(err) : "Unknown dlerror";
    }
#endif

namespace ormi::cpu {
    typedef void (*FusedKernel)(int, float*, const float**);
    static std::unordered_map<std::uint64_t, FusedKernel> kernel_cache;
    static std::mutex kernel_cache_mutex;

    namespace {
        struct JITArtifactPaths {
            std::filesystem::path cpp_file;
            std::filesystem::path dll_file;
        };

        bool file_contents_equal(const std::filesystem::path& path, const std::string& expected) {
            std::ifstream in(path, std::ios::binary);
            if (!in) {
                return false;
            }

            std::ostringstream ss;
            ss << in.rdbuf();

            return ss.str() == expected;
        }

        JITArtifactPaths resolve_jit_artifact_paths(
            const std::filesystem::path& cache_dir,
            std::uint64_t code_hash,
            const std::string& kernel_code
        ) {
            for (int attempt = 0; attempt < 1024; ++attempt) {
                std::string suffix =
                    attempt == 0
                        ? std::to_string(code_hash)
                        : std::to_string(code_hash) + "_" + std::to_string(attempt);

                std::filesystem::path cpp_file =
                    cache_dir / ("ormi_jit_" + suffix + ".cpp");

                std::filesystem::path dll_file =
                    cache_dir / ("ormi_jit_" + suffix + OS_LIB_EXT);

                if (!std::filesystem::exists(cpp_file)) {
                    return {cpp_file, dll_file};
                }

                if (file_contents_equal(cpp_file, kernel_code)) {
                    return {cpp_file, dll_file};
                }
            }

            ORMI_FAIL("CPU JIT cache collision resolution failed after 1024 attempts.");
        }

        std::uint64_t stable_hash(const std::string& text) {
            std::uint64_t hash = 1469598103934665603ull;

            for (unsigned char c : text) {
                hash ^= c;
                hash *= 1099511628211ull;
            }

            return hash;
        }

        std::filesystem::path cpu_jit_cache_dir() {
            const char* env = std::getenv("ORMI_CPU_JIT_CACHE_DIR");

            std::filesystem::path dir =
                (env && *env)
                    ? std::filesystem::path(env)
                    : (std::filesystem::temp_directory_path() / "ormi" / "cpu_jit");

            std::filesystem::create_directories(dir);
            return std::filesystem::absolute(dir);
        }

        std::string quote_path(const std::filesystem::path& path) {
            std::string s = path.string();
            std::string out = "\"";

            for (char c : s) {
                if (c == '"') {
                    out += "\\\"";
                } else {
                    out += c;
                }
            }

            out += "\"";
            return out;
        }
    }

    void execute_jit(const std::shared_ptr<Node>& root) {
        CompilationResult ast = compile(root);
        if (!root->data) {
            root->data = empty(ast.out_shape, Device::cpu(), root->dtype);
            root->view.shape = ast.out_shape;
            root->view.strides = calc_contiguous_strides(ast.out_shape);
        }

        float* out_ptr = static_cast<float*>(root->data->ptr);

        std::vector<const float*> leaves;
        std::vector<bool> is_scalar;
        for (auto& node : ast.leaf_nodes) {
            leaves.push_back(static_cast<const float*>(node->data->ptr));
            is_scalar.push_back(node->view.shape.empty() || (node->data->bytes / sizeof(float) == 1));
        }

        int ndim = ast.out_shape.size();
        std::vector<int> out_strides = root->view.strides;
        std::vector<std::vector<int>> leaf_strides(leaves.size(), std::vector<int>(ndim, 0));

        for (size_t L = 0; L < leaves.size(); ++L) {
            if (is_scalar[L]) continue;
            const auto& leaf_view = ast.leaf_nodes[L]->view;
            int offset = ndim - leaf_view.shape.size();
            for (int d = 0; d < ndim; ++d) {
                int original_dim_idx = d - offset;
                if (original_dim_idx >= 0 && leaf_view.shape[original_dim_idx] != 1) {
                    leaf_strides[L][d] = leaf_view.strides[original_dim_idx];
                } else {
                    leaf_strides[L][d] = 0;
                }
            }
        }

        //
        // Code gen
        //

        std::stringstream src;
        src << "#include <cmath>\n";

        // Cross-platform compiler export directives
        src << "#ifdef _WIN32\n";
        src << "  #define ORMI_EXPORT __declspec(dllexport)\n";
        src << "#else\n";
        src << "  #define ORMI_EXPORT __attribute__((visibility(\"default\")))\n";
        src << "#endif\n";

        src << "extern \"C\" ORMI_EXPORT void fused_kernel(int max_elements, float* OUT, const float** IN) {\n";
        src << "    for (int i = 0; i < max_elements; ++i) {\n";

        for (int r = 0; r < ast.reg_count; ++r) {
            src << "        float reg_" << r << " = 0.0f;\n";
        }

        // Stride/load logic
        for (size_t L = 0; L < leaves.size(); ++L) {
            if (is_scalar[L]) {
                src << "        reg_" << ast.leaf_regs[L] << " = IN[" << L << "][0];\n";
            } else {
                src << "        int leaf_idx_" << L << " = 0;\n";
                src << "        int temp_" << L << " = i;\n";
                for (int d = 0; d < ndim; ++d) {
                    src << "        leaf_idx_" << L << " += (temp_" << L << " / " << out_strides[d] << ") * " << leaf_strides[L][d] << ";\n";
                    src << "        temp_" << L << " %= " << out_strides[d] << ";\n";
                }
                src << "        reg_" << ast.leaf_regs[L] << " = IN[" << L << "][leaf_idx_" << L << "];\n";
            }
        }

        // Lower the tape
        for (const auto& inst : ast.tape) {
            auto string_builder = CPULoweringRegistry::get(inst.op);
            std::vector<std::string> args = {
                "reg_" + std::to_string(inst.src1),
                "reg_" + std::to_string(inst.src2)
            };
            src << "        reg_" << inst.dest << " = " << string_builder(args) << "\n";
        }

        src << "        OUT[i] = reg_" << ast.root_reg << ";\n";
        src << "    }\n";
        src << "}\n";

        //
        // Compile
        //

        std::string kernel_code = src.str();

        const char* opt_env = std::getenv("ORMI_CPU_JIT_OPT");
        std::string opt_flag = opt_env ? std::string(opt_env) : "-O3";

        if (
            opt_flag != "-O0" &&
            opt_flag != "-O1" &&
            opt_flag != "-O2" &&
            opt_flag != "-O3" &&
            opt_flag != "-Ofast"
        ) {
            ORMI_THROW(
                RuntimeError,
                "Invalid ORMI_CPU_JIT_OPT. Expected -O0, -O1, -O2, -O3, or -Ofast. Got: " + opt_flag
            );
        }

        std::uint64_t code_hash = stable_hash(kernel_code + "\n// ORMI_CPU_JIT_OPT=" + opt_flag);

        FusedKernel kernel = nullptr;

        {
            std::lock_guard<std::mutex> lock(kernel_cache_mutex);

            auto cached = kernel_cache.find(code_hash);

            if (cached == kernel_cache.end()) {
                std::filesystem::path cache_dir = cpu_jit_cache_dir();

                auto paths = resolve_jit_artifact_paths(cache_dir, code_hash, kernel_code);
                std::filesystem::path cpp_file = paths.cpp_file;
                std::filesystem::path dll_file = paths.dll_file;

                if (!std::filesystem::exists(dll_file)) {
                    std::ofstream out(cpp_file);

                    if (!out) {
                        ORMI_FAIL("Failed to write CPU JIT source file: " + cpp_file.string());
                    }

                    out << kernel_code;
                    out.close();

                    std::string cmd =
                        "g++ " + opt_flag +
                        " -shared -fPIC " +
                        quote_path(cpp_file) +
                        " -o " +
                        quote_path(dll_file);

                    int ret = std::system(cmd.c_str());

                    if (ret != 0) {
                        ORMI_FAIL(
                            "CPU JIT compilation failed. Ensure g++ is in your PATH. Command: " + cmd
                        );
                    }
                }

                void* hLib = load_shared_library(dll_file.string());
                if (!hLib) {
                    ORMI_FAIL("Failed to load compiled JIT library: " + get_dl_error());
                }

                void* symbol = get_symbol(hLib, "fused_kernel");
                if (!symbol) {
                    ORMI_FAIL("Failed to locate 'fused_kernel' symbol inside binary: " + get_dl_error());
                }

                kernel = reinterpret_cast<FusedKernel>(symbol);
                kernel_cache[code_hash] = kernel;
            } else {
                kernel = cached->second;
            }
        }

        kernel(ast.max_elements, out_ptr, leaves.data());
    }
}