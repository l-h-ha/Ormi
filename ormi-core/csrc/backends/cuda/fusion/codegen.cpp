#include <algorithm>
#include <cstddef>
#include <sstream>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/kind.h>
#include <ormi/core/tensor.h>

#include "backends/cuda/fusion/codegen.h"
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

        std::string_view value_type(core::DType dtype) {
            return dtype == core::DType::float64 ? "double" : "float";
        }

        std::string_view literal(
            core::DType dtype,
            core::runtime::fusion::Constant constant
        ) {
            if (dtype != core::DType::float64) {
                return constant == core::runtime::fusion::Constant::zero ? "0.0f" : "1.0f";
            }

            return constant == core::runtime::fusion::Constant::zero ? "0.0" : "1.0";
        }

        std::string unary(
            core::DType dtype,
            core::ops::UnaryOp op,
            std::string_view input
        ) {
            const bool single = dtype != core::DType::float64;

            switch (op) {
                case core::ops::UnaryOp::neg:
                    return "-" + std::string(input);
                case core::ops::UnaryOp::exp:
                    return std::string(single ? "expf(" : "exp(") + std::string(input) + ")";
                case core::ops::UnaryOp::log:
                    return std::string(single ? "logf(" : "log(") + std::string(input) + ")";
                case core::ops::UnaryOp::sin:
                    return std::string(single ? "sinf(" : "sin(") + std::string(input) + ")";
                case core::ops::UnaryOp::cos:
                    return std::string(single ? "cosf(" : "cos(") + std::string(input) + ")";
            }

            ORMI_THROW(
                OperationError,
                "CUDA fusion received an unknown unary operation."
            );
        }

        std::string_view binary_operator(core::ops::BinaryOp op) {
            switch (op) {
                case core::ops::BinaryOp::add:
                    return "+";
                case core::ops::BinaryOp::sub:
                    return "-";
                case core::ops::BinaryOp::mul:
                    return "*";
                case core::ops::BinaryOp::div:
                    return "/";
            }

            ORMI_THROW(OperationError, "CUDA fusion received an unknown binary operation.");
        }

        std::string binary(
            core::ops::BinaryOp op,
            std::string_view lhs,
            std::string_view rhs
        ) {
            return std::string(lhs) + " " + std::string(binary_operator(op)) + " " +
                std::string(rhs);
        }

        std::string value(core::runtime::fusion::Index index) {
            return "v" + std::to_string(index);
        }

        std::string offset(
            std::string_view index,
            const std::vector<core::runtime::fusion::Axis>& axes
        );

        std::string rounded(core::DType dtype, std::string expression) {
            if (dtype == core::DType::float16) {
                return "ormi_half_round(" + expression + ")";
            }
            return expression;
        }

        std::string input_value(
            core::DType dtype,
            const core::runtime::fusion::Input& input,
            const Tile* tile,
            bool physical_order
        ) {
            std::string index;
            if (tile) {
                const auto& strides = tile->inputs[input.index];
                for (
                    std::size_t dimension = 0;
                    dimension < strides.size();
                    ++dimension
                ) {
                    if (strides[dimension] == 0) {
                        continue;
                    }
                    if (!index.empty()) {
                        index += " + ";
                    }
                    index += "index" + std::to_string(dimension);
                    if (strides[dimension] != 1) {
                        index += " * " +
                            std::to_string(strides[dimension]);
                    }
                }
                if (index.empty()) {
                    index = "0";
                }
            } else if (physical_order) {
                index = "reduced_index";
            } else {
                index = input.linear ? "i" : offset("i", input.axes);
            }

            auto expression =
                "input" + std::to_string(input.index) + "[" + index + "]";
            if (dtype == core::DType::float16) {
                return "ormi_half_value(" + expression + ")";
            }
            return expression;
        }

        std::string output_value(core::DType dtype, std::string_view expression) {
            if (dtype == core::DType::float16) {
                return "ormi_half_bits(" + std::string(expression) + ")";
            }
            return std::string(expression);
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

        std::string cast_value(core::DType dtype, std::string_view expression) {
            switch (dtype) {
                case core::DType::float64:
                    return "static_cast<double>(" + std::string(expression) + ")";
                case core::DType::float32:
                    return "static_cast<float>(" + std::string(expression) + ")";
                case core::DType::float16:
                    return "ormi_half_round(" + std::string(expression) + ")";
                case core::DType::int32:
                case core::DType::boolean:
                    ORMI_THROW(DTypeError, "CUDA fusion cannot cast to a non-floating dtype.");
            }

            ORMI_THROW(DTypeError, "CUDA fusion received an unknown cast dtype.");
        }

        bool uses_float16(const core::runtime::fusion::Expr& expression) {
            for (const auto& node : expression.nodes) {
                if (node.dtype == core::DType::float16) {
                    return true;
                }
            }
            return false;
        }

        std::string offset(
            std::string_view index,
            const std::vector<core::runtime::fusion::Axis>& axes
        ) {
            if (axes.empty()) {
                return "0";
            }

            std::ostringstream result;
            for (std::size_t i = 0; i < axes.size(); ++i) {
                const auto& axis = axes[i];
                if (i != 0) {
                    result << " + ";
                }

                result << "(";
                if (axis.divisor != 1) {
                    result << "(" << index << " / " << axis.divisor << ")";
                } else {
                    result << index;
                }
                result << " % " << axis.extent << ")";
                if (axis.stride != 1) {
                    result << " * " << axis.stride;
                }
            }
            return result.str();
        }

        void require_prior(core::runtime::fusion::Index referenced, std::size_t current) {
            if (referenced >= current) {
                ORMI_THROW(OperationError, "Fusion expression nodes must reference prior nodes.");
            }
        }

        std::string reduced_value(
            core::DType dtype,
            const core::runtime::fusion::Reduction& reduction
        ) {
            if (reduction.op == core::ops::ReduceOp::sum) {
                return "total";
            }

            return "total / static_cast<" + std::string(value_type(dtype)) + ">(" +
                std::to_string(reduction.reduced.size) + "ull)";
        }

        void emit_nodes(
            std::ostringstream& source,
            const core::runtime::fusion::Expr& expression,
            const Tile* tile = nullptr,
            bool physical_order = false
        ) {
            for (std::size_t i = 0; i < expression.nodes.size(); ++i) {
                const auto& node = expression.nodes[i];
                source << "        const " << value_type(node.dtype) << " "
                       << value(static_cast<core::runtime::fusion::Index>(i)) << " = ";
                const auto& instruction = node.instruction;

                if (
                    const auto* input =
                        std::get_if<core::runtime::fusion::Input>(&instruction)
                ) {
                    if (input->index >= expression.inputs.size()) {
                        ORMI_THROW(OperationError, "Fusion expression references an invalid input.");
                    }
                    source << input_value(
                        node.dtype,
                        *input,
                        tile,
                        physical_order
                    );
                } else if (
                    const auto* constant =
                        std::get_if<core::runtime::fusion::Constant>(&instruction)
                ) {
                    source << literal(node.dtype, *constant);
                } else if (
                    const auto* cast_node =
                        std::get_if<core::runtime::fusion::Cast>(&instruction)
                ) {
                    require_prior(cast_node->input, i);
                    source << cast_value(node.dtype, value(cast_node->input));
                } else if (
                    const auto* unary_node =
                        std::get_if<core::runtime::fusion::Unary>(&instruction)
                ) {
                    require_prior(unary_node->input, i);
                    const auto operation = unary(
                        node.dtype,
                        unary_node->op,
                        value(unary_node->input)
                    );
                    source << (i == expression.output && !expression.reduction
                        ? operation
                        : rounded(node.dtype, operation));
                } else if (
                    const auto* binary_node =
                        std::get_if<core::runtime::fusion::Binary>(&instruction)
                ) {
                    require_prior(binary_node->lhs, i);
                    require_prior(binary_node->rhs, i);
                    const auto operation = binary(
                        binary_node->op,
                        value(binary_node->lhs),
                        value(binary_node->rhs)
                    );
                    source << (i == expression.output && !expression.reduction
                        ? operation
                        : rounded(node.dtype, operation));
                }

                source << ";\n";
            }
        }

        void emit_coordinates(
            std::ostringstream& source,
            const core::runtime::fusion::Expr& expression,
            const std::vector<std::size_t>& dimensions,
            std::string_view flat,
            std::string_view indent
        ) {
            std::size_t stride = 1;
            for (const auto dimension : dimensions) {
                stride *= expression.shape[dimension];
            }

            for (std::size_t i = 0; i < dimensions.size(); ++i) {
                const auto dimension = dimensions[i];
                const auto extent = expression.shape[dimension];
                stride /= extent;
                source
                    << indent << "const unsigned long long index"
                    << dimension << " = ";
                if (extent == 1) {
                    source << "0";
                } else if (dimensions.size() == 1) {
                    source << flat;
                } else if (i == 0) {
                    source << flat << " / " << stride << "ull";
                } else if (stride == 1) {
                    source << flat << " % " << extent << "ull";
                } else {
                    source << "(" << flat << " / " << stride
                        << "ull) % " << extent << "ull";
                }
                source << ";\n";
            }
        }

        std::string tiled_output_offset(
            const core::runtime::fusion::Expr& expression,
            const Tile& tile
        ) {
            auto stride = expression.reduction->output.size;
            std::string result;
            for (const auto dimension : tile.output) {
                stride /= expression.shape[dimension];
                if (!result.empty()) {
                    result += " + ";
                }
                result += "index" + std::to_string(dimension);
                if (stride != 1) {
                    result += " * " + std::to_string(stride);
                }
            }
            return result;
        }

        void emit_tiled_reduction(
            std::ostringstream& source,
            const core::runtime::fusion::Expr& expression,
            const Tile& tile
        ) {
            const auto& reduction = *expression.reduction;
            const auto scalar = value_type(expression.dtype);
            std::vector<std::size_t> outer;
            for (const auto dimension : tile.output) {
                if (dimension != tile.axis) {
                    outer.push_back(dimension);
                }
            }
            const auto extent = expression.shape[tile.axis];
            const auto tiles =
                (extent + tile.width - 1) / tile.width;

            source
                << ",\n    unsigned long long parts\n"
                << ") {\n"
                << "    (void)parts;\n"
                << "    for (unsigned long long task = blockIdx.x; task < "
                << tile.tasks << "ull; task += gridDim.x) {\n"
                << "        const unsigned long long tile_index = task % "
                << tiles << "ull;\n"
                << "        const unsigned long long outer_index = task / "
                << tiles << "ull;\n";
            emit_coordinates(
                source,
                expression,
                outer,
                "outer_index",
                "        "
            );
            source
                << "        const unsigned long long index" << tile.axis
                << " = tile_index * " << tile.width
                << "ull + threadIdx.x;\n"
                << "        if (index" << tile.axis << " >= " << extent
                << "ull) {\n"
                << "            continue;\n"
                << "        }\n"
                << "        " << scalar << " total = "
                << literal(
                    expression.dtype,
                    core::runtime::fusion::Constant::zero
                )
                << ";\n"
                << "        for (unsigned long long reduced_index = 0; "
                << "reduced_index < " << reduction.reduced.size
                << "ull; ++reduced_index) {\n";
            emit_coordinates(
                source,
                expression,
                tile.reduced,
                "reduced_index",
                "            "
            );
            emit_nodes(source, expression, &tile);
            source
                << "            total += " << value(expression.output)
                << ";\n"
                << "        }\n"
                << "        output[" << tiled_output_offset(expression, tile)
                << "] = "
                << output_value(
                    expression.dtype,
                    reduced_value(expression.dtype, reduction)
                )
                << ";\n"
                << "    }\n"
                << "}\n";
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
        source
            << "extern \"C\" __global__ void ormi_kernel(\n"
            << "    "
            << (expression.reduction && !tiled ? scalar : storage)
            << "* output";

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

        if (tiled) {
            emit_tiled_reduction(source, expression, *tiled);
        } else if (expression.reduction) {
            const auto& reduction = *expression.reduction;
            source
                << ",\n    unsigned long long parts\n"
                << ") {\n"
                << "    const unsigned long long tasks = "
                << reduction.output.size << "ull * parts;\n"
                << "    for (unsigned long long task = blockIdx.x; task < tasks; task += gridDim.x) {\n"
                << "        const unsigned long long output_index = task / parts;\n"
                << "        const unsigned long long part = task % parts;\n"
                << "        const unsigned long long input_base = "
                << offset("output_index", reduction.output.axes) << ";\n"
                << "        " << scalar << " total = "
                << literal(expression.dtype, core::runtime::fusion::Constant::zero)
                << ";\n\n"
                << "        for (unsigned long long reduced_index =\n"
                << "                 part * blockDim.x + threadIdx.x;\n"
                << "             reduced_index < " << reduction.reduced.size << "ull;\n"
                << "             reduced_index += parts * blockDim.x) {\n"
                << "            const unsigned long long i = input_base + "
                << offset("reduced_index", reduction.reduced.axes) << ";\n";

            emit_nodes(source, expression, nullptr, physical);

            source
                << "            total += " << value(expression.output) << ";\n"
                << "        }\n\n"
                << "        const auto block = ormi_block_sum(total);\n"
                << "        if (threadIdx.x == 0) {\n"
                << "            output[task] = block;\n"
                << "        }\n"
                << "    }\n\n"
                << "}\n\n"
                << "extern \"C\" __global__ void ormi_finish(\n"
                << "    " << storage << "* output,\n"
                << "    const " << scalar << "* partials,\n"
                << "    unsigned long long parts\n"
                << ") {\n"
                << "    for (unsigned long long output_index = blockIdx.x;\n"
                << "         output_index < " << reduction.output.size << "ull;\n"
                << "         output_index += gridDim.x) {\n"
                << "        " << scalar << " total = "
                << literal(expression.dtype, core::runtime::fusion::Constant::zero)
                << ";\n"
                << "        for (unsigned long long part = threadIdx.x; part < parts; part += blockDim.x) {\n"
                << "            total += partials[output_index * parts + part];\n"
                << "        }\n"
                << "        total = ormi_block_sum(total);\n"
                << "        if (threadIdx.x == 0) {\n"
                << "            output[output_index] = "
                << output_value(
                    expression.dtype,
                    reduced_value(expression.dtype, reduction)
                ) << ";\n"
                << "        }\n"
                << "    }\n"
                << "}\n";
        } else {
            source
                << ",\n    unsigned long long elements\n"
                << ") {\n"
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
