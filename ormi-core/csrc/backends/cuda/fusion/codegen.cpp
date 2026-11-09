#include <algorithm>
#include <cstddef>
#include <sstream>
#include <string>
#include <string_view>

#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/kind.h>
#include <ormi/core/tensor.h>

#include "backends/cuda/fusion/codegen.h"
#include "backends/cuda/fusion/codegen/emit.h"
#include "backends/cuda/fusion/order.h"
#include "backends/cuda/fusion/tile.h"

namespace ormi::backends::cuda::fusion {
    namespace {
        std::string_view storage_type(core::DType dtype) {
            switch (dtype) {
                case core::DType::float64:
                    return "double";
                case core::DType::float32:
                    return "float";
                case core::DType::float16:
                    return "unsigned short";
                case core::DType::int32:
                case core::DType::boolean:
                    ORMI_THROW(DTypeError, "CUDA fusion requires a floating-point dtype.");
            }

            ORMI_THROW(DTypeError, "CUDA fusion received an unknown dtype.");
        }

        std::size_t output_count(
            const core::runtime::fusion::Expr& expression
        ) {
            return 1 + expression.auxiliary_outputs.size();
        }

        core::runtime::fusion::Index output_index(
            const core::runtime::fusion::Expr& expression,
            std::size_t output
        ) {
            return output == 0
                ? expression.output
                : expression.auxiliary_outputs[output - 1];
        }

        std::string output_name(std::size_t output) {
            return output == 0
                ? "output"
                : "output" + std::to_string(output);
        }

        void emit_signature(
            std::ostringstream& source,
            const core::runtime::fusion::Expr& expression,
            std::string_view output_type
        ) {
            source
                << "extern \"C\" __global__ void ormi_kernel(\n    "
                << output_type << "* output";

            for (
                std::size_t output = 1;
                output < output_count(expression);
                ++output
            ) {
                const auto index = output_index(expression, output);
                source
                    << ",\n    "
                    << storage_type(expression.nodes[index].dtype)
                    << "* " << output_name(output);
            }

            for (std::size_t i = 0; i < expression.inputs.size(); ++i) {
                source << ",\n    const " << storage_type(expression.inputs[i]->dtype())
                       << "* input" << i;
            }

            source
                << (expression.reduction
                    ? ",\n    unsigned long long parts\n) {\n"
                    : ",\n    unsigned long long elements\n) {\n");
        }

        void emit_pointwise_stores(
            std::ostringstream& source,
            const core::runtime::fusion::Expr& expression
        ) {
            for (
                std::size_t output = 0;
                output < output_count(expression);
                ++output
            ) {
                const auto index = output_index(expression, output);
                source
                    << "        " << output_name(output) << "[i] = "
                    << output_value(
                        expression.nodes[index].dtype,
                        value(index)
                    )
                    << ";\n";
            }
        }

        bool uses_float16(const core::runtime::fusion::Expr& expression) {
            for (const auto& node : expression.nodes) {
                if (node.dtype == core::DType::float16) {
                    return true;
                }
            }
            return false;
        }

        void emit_support(
            std::ostringstream& source,
            const core::runtime::fusion::Expr& expression,
            bool tiled
        ) {
            if (uses_float16(expression)) {
                source << R"(
namespace {
    __device__ __forceinline__ unsigned short ormi_half_bits(float value) {
        unsigned short result;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(value));
        return result;
    }

    __device__ __forceinline__ float ormi_half_value(unsigned short value) {
        float result;
        asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(value));
        return result;
    }

    __device__ __forceinline__ float ormi_half_round(float value) {
        return ormi_half_value(ormi_half_bits(value));
    }
}

)";
            }

            if (expression.reduction && !tiled) {
                source
                    << "template <typename T>\n"
                    << "__device__ T ormi_block_sum(T value) {\n"
                    << "    extern __shared__ unsigned char shared_bytes[];\n"
                    << "    auto* partial = reinterpret_cast<T*>(shared_bytes);\n"
                    << "    partial[threadIdx.x] = value;\n"
                    << "    __syncthreads();\n\n"
                    << "    for (unsigned int width = blockDim.x / 2; width > 0; width /= 2) {\n"
                    << "        if (threadIdx.x < width) {\n"
                    << "            partial[threadIdx.x] += partial[threadIdx.x + width];\n"
                    << "        }\n"
                    << "        __syncthreads();\n"
                    << "    }\n"
                    << "    return partial[0];\n"
                    << "}\n\n";
            }
        }
    }

    std::string emit(const core::runtime::fusion::Expr& expression) {
        if (
            expression.nodes.empty() ||
            expression.output >= expression.nodes.size() ||
            std::any_of(
                expression.auxiliary_outputs.begin(),
                expression.auxiliary_outputs.end(),
                [&expression](const auto output) {
                    return output >= expression.nodes.size();
                }
            )
        ) {
            ORMI_THROW(
                OperationError,
                "Cannot generate CUDA code for a malformed fusion expression."
            );
        }

        if (expression.reduction && !expression.auxiliary_outputs.empty()) {
            ORMI_THROW(
                OperationError,
                "CUDA fusion does not support auxiliary reduction outputs."
            );
        }

        if (expression.reduction &&
            expression.reduction->op != core::ops::ReduceOp::sum &&
            expression.reduction->op != core::ops::ReduceOp::mean) {
            ORMI_THROW(OperationError, "CUDA fusion received an unsupported reduction.");
        }

        const auto storage = storage_type(expression.dtype);
        const auto scalar = value_type(expression.dtype);
        const auto tiled = tile(expression);
        const auto physical = physical_order(expression);
        std::ostringstream source;
        emit_support(source, expression, tiled.has_value());
        emit_signature(source, expression, expression.reduction && !tiled ? scalar : storage);

        if (tiled) {
            emit_tiled_reduction(source, expression, *tiled);
        } else if (expression.reduction) {
            emit_reduction(source, expression, storage, scalar, physical);
        } else {
            source
                << "    const unsigned long long start =\n"
                << "        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;\n"
                << "    const unsigned long long stride =\n"
                << "        static_cast<unsigned long long>(blockDim.x) * gridDim.x;\n\n"
                << "    for (unsigned long long i = start; i < elements; i += stride) {\n";

            emit_nodes(source, expression);

            emit_pointwise_stores(source, expression);
            source << "    }\n}\n";
        }
        return source.str();
    }
}
