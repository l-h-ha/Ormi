#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <nvrtc.h>

#include <ormi/core/error.h>

#include "backends/cuda/fusion/nvrtc.h"

namespace ormi::backends::cuda::fusion::nvrtc {
    namespace {
        std::vector<int> supported_architectures() {
            int count = 0;
            auto status = nvrtcGetNumSupportedArchs(&count);

            if (status != NVRTC_SUCCESS) {
                ORMI_THROW(
                    jitError,
                    std::string("nvrtcGetNumSupportedArchs failed: ") +
                        nvrtcGetErrorString(status)
                );
            }

            if (count <= 0) {
                ORMI_THROW(jitError, "NVRTC reports no supported GPU architectures.");
            }

            std::vector<int> architectures(static_cast<std::size_t>(count));
            status = nvrtcGetSupportedArchs(architectures.data());

            if (status != NVRTC_SUCCESS) {
                ORMI_THROW(
                    jitError,
                    std::string("nvrtcGetSupportedArchs failed: ") +
                        nvrtcGetErrorString(status)
                );
            }

            return architectures;
        }

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

            std::string compile(int selected_target) {
                const auto architecture =
                    "--gpu-architecture=compute_" + std::to_string(selected_target);
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
    }

    int target(int major, int minor) {
        if (major <= 0 || minor < 0 || minor > 9) {
            ORMI_THROW(jitError, "Invalid CUDA compute capability.");
        }

        static const auto architectures = supported_architectures();
        const auto requested = static_cast<std::int64_t>(major) * 10 + minor;
        int selected = 0;

        for (const auto architecture : architectures) {
            if (architecture <= requested && architecture > selected) {
                selected = architecture;
            }
        }

        if (selected == 0) {
            ORMI_THROW(
                jitError,
                "NVRTC has no supported PTX target for compute_" +
                    std::to_string(requested) + "."
            );
        }

        return selected;
    }

    std::string context(int major, int minor) {
        const auto selected_target = target(major, minor);
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
            "compute_" + std::to_string(selected_target) +
            "\nnvrtc_" + std::to_string(nvrtc_major) +
            "." + std::to_string(nvrtc_minor);
    }

    std::string compile(const std::string& source, int major, int minor) {
        Program program(source);
        return program.compile(target(major, minor));
    }
}
