#include <cstddef>
#include <sstream>
#include <string>
#include <string_view>
#include <variant>

#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/kind.h>

#include "backends/cpu/jit/codegen.h"

namespace ormi::backends::cpu::jit {
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
                    ORMI_THROW(DTypeError, "CPU fusion code generation currently supports Float64 and Float32.");
            }

            ORMI_THROW(DTypeError, "CPU fusion code generation received an unknown dtype.");
        }

        std::string_view literal(core::DType dtype, core::runtime::fusion::Constant constant) {
            if (dtype == core::DType::float32) {
                return constant == core::runtime::fusion::Constant::zero ? "0.0f" : "1.0f";
            }

            return constant == core::runtime::fusion::Constant::zero ? "0.0" : "1.0";
        }

        std::string unary(core::ops::UnaryOp op, std::string_view input) {
            switch (op) {
                case core::ops::UnaryOp::neg:
                    return "-" + std::string(input);
                case core::ops::UnaryOp::exp:
                    return "std::exp(" + std::string(input) + ")";
                case core::ops::UnaryOp::log:
                    return "std::log(" + std::string(input) + ")";
                case core::ops::UnaryOp::sin:
                    return "std::sin(" + std::string(input) + ")";
                case core::ops::UnaryOp::cos:
                    return "std::cos(" + std::string(input) + ")";
            }

            ORMI_THROW(OperationError, "CPU fusion code generation received an unknown unary operation.");
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

            ORMI_THROW(OperationError, "CPU fusion code generation received an unknown binary operation.");
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
        if (expression.nodes.empty() || expression.output >= expression.nodes.size()) {
            ORMI_THROW(OperationError, "Cannot generate code for an empty or malformed fusion expression.");
        }

        const auto scalar = type(expression.dtype);
        std::ostringstream source;
        // TODO: Add explicit aliasing, alignment, SIMD, and vector-math policies after
        // strict and relaxed floating-point modes are part of the public contract.
        source
            << "#include <cmath>\n\n"
            << "#if defined(_WIN32)\n"
            << "#define ORMI_EXPORT __declspec(dllexport)\n"
            << "#else\n"
            << "#define ORMI_EXPORT __attribute__((visibility(\"default\")))\n"
            << "#endif\n\n"
            << "extern \"C\" ORMI_EXPORT void ormi_kernel(\n"
            << "    void* output_ptr,\n"
            << "    const void* const* input_ptrs,\n"
            << "    unsigned long long elements\n"
            << ") {\n"
            << "    auto* output = static_cast<" << scalar << "*>(output_ptr);\n";

        for (std::size_t i = 0; i < expression.inputs.size(); ++i) {
            source
                << "    const auto* input" << i << " = static_cast<const "
                << scalar << "*>(input_ptrs[" << i << "]);\n";
        }

        source << "\n    for (unsigned long long i = 0; i < elements; ++i) {\n";

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
                source << unary(unary_node->op, value(unary_node->input));

            } else if (const auto* binary_node = std::get_if<core::runtime::fusion::Binary>(&node)) {
                require_prior(binary_node->lhs, i);
                require_prior(binary_node->rhs, i);
                source
                    << value(binary_node->lhs) << " "
                    << binary(binary_node->op) << " "
                    << value(binary_node->rhs);
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
