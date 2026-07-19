#include <cstddef>
#include <sstream>
#include <string>
#include <string_view>
#include <variant>

#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/kind.h>

#include "backends/cuda/fusion/codegen.h"

namespace ormi::backends::cuda::fusion {
    namespace {
        std::string_view type(core::DType dtype) {
            switch (dtype) {
                case core::DType::float64:
                    return "double";
                case core::DType::float32:
                    return "float";
                case core::DType::float16:
                case core::DType::int32:
                case core::DType::boolean:
                    ORMI_THROW(DTypeError, "CUDA fusion supports Float64 and Float32.");
            }

            ORMI_THROW(DTypeError, "CUDA fusion received an unknown dtype.");
        }

        std::string_view literal(
            core::DType dtype,
            core::runtime::fusion::Constant constant
        ) {
            if (dtype == core::DType::float32) {
                return constant == core::runtime::fusion::Constant::zero ? "0.0f" : "1.0f";
            }

            return constant == core::runtime::fusion::Constant::zero ? "0.0" : "1.0";
        }

        std::string unary(
            core::DType dtype,
            core::ops::UnaryOp op,
            std::string_view input
        ) {
            const bool single = dtype == core::DType::float32;

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

        std::string_view binary(core::ops::BinaryOp op) {
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

        std::string value(core::runtime::fusion::Index index) {
            return "v" + std::to_string(index);
        }

        void require_prior(core::runtime::fusion::Index referenced, std::size_t current) {
            if (referenced >= current) {
                ORMI_THROW(OperationError, "Fusion expression nodes must reference prior nodes.");
            }
        }
    }

    std::string emit(const core::runtime::fusion::Expr& expression) {
        if (
            expression.nodes.empty() ||
            expression.output >= expression.nodes.size()
        ) {
            ORMI_THROW(
                OperationError,
                "Cannot generate CUDA code for a malformed fusion expression."
            );
        }

        const auto scalar = type(expression.dtype);
        std::ostringstream source;
        source
            << "extern \"C\" __global__ void ormi_kernel(\n"
            << "    " << scalar << "* output";

        for (std::size_t i = 0; i < expression.inputs.size(); ++i) {
            source << ",\n    const " << scalar << "* input" << i;
        }

        source
            << ",\n    unsigned long long elements\n"
            << ") {\n"
            << "    const unsigned long long start =\n"
            << "        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;\n"
            << "    const unsigned long long stride =\n"
            << "        static_cast<unsigned long long>(blockDim.x) * gridDim.x;\n\n"
            << "    for (unsigned long long i = start; i < elements; i += stride) {\n";

        for (std::size_t i = 0; i < expression.nodes.size(); ++i) {
            source << "        const " << scalar << " "
                   << value(static_cast<core::runtime::fusion::Index>(i)) << " = ";
            const auto& node = expression.nodes[i];

            if (const auto* input = std::get_if<core::runtime::fusion::Input>(&node)) {
                if (input->index >= expression.inputs.size()) {
                    ORMI_THROW(OperationError, "Fusion expression references an invalid input.");
                }
                source << "input" << input->index << "[i]";
            } else if (const auto* constant = std::get_if<core::runtime::fusion::Constant>(&node)) {
                source << literal(expression.dtype, *constant);
            } else if (const auto* unary_node = std::get_if<core::runtime::fusion::Unary>(&node)) {
                require_prior(unary_node->input, i);
                source << unary(expression.dtype, unary_node->op, value(unary_node->input));
            } else if (
                const auto* binary_node =
                    std::get_if<core::runtime::fusion::Binary>(&node)
            ) {
                require_prior(binary_node->lhs, i);
                require_prior(binary_node->rhs, i);
                source << value(binary_node->lhs) << " " << binary(binary_node->op)
                       << " " << value(binary_node->rhs);
            }

            source << ";\n";
        }

        source
            << "        output[i] = " << value(expression.output) << ";\n"
            << "    }\n"
            << "}\n";
        return source.str();
    }
}
