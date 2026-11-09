#include <cstddef>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <ormi/core/error.h>

#include "backends/cpu/jit/codegen/input.h"

namespace ormi::backends::cpu::jit::codegen {
    namespace {
        std::string input_offset(
            const core::runtime::fusion::Input& input
        ) {
            if (input.linear) {
                return "i";
            }
            return offset("i", input.axes);
        }

        std::string packed_strided_input(
            const core::runtime::fusion::Input& input,
            const Loop& loop
        ) {
            const auto name = "input" + std::to_string(input.index);
            const auto& strides = loop.strides[input.index];
            const auto inner = loop.reduced_dimensions.back();
            std::string address;

            for (
                std::size_t dimension = 0;
                dimension < strides.size();
                ++dimension
            ) {
                const auto stride = strides[dimension];
                if (stride == 0) {
                    continue;
                }
                if (!address.empty()) {
                    address += " + ";
                }
                address += dimension == inner
                    ? "inner_index"
                    : "index" + std::to_string(dimension);
                if (stride != 1) {
                    address += " * " + std::to_string(stride);
                }
            }

            if (strides[inner] == 0) {
                return "_mm256_set1_ps(" + name + "[" +
                    (address.empty() ? "0" : address) + "])";
            }
            if (strides[inner] == 1) {
                return "_mm256_loadu_ps(" + name + " + " + address + ")";
            }

            ORMI_THROW(
                OperationError,
                "Packed strided fusion requires unit or broadcast inner strides."
            );
        }
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

    std::string input_value(
        core::DType dtype,
        const core::runtime::fusion::Input& input,
        const Loop& loop
    ) {
        auto expression = "input" + std::to_string(input.index);
        if (loop.strided) {
            expression += "[";
            bool first = true;
            for (
                std::size_t dimension = 0;
                dimension < loop.strides[input.index].size();
                ++dimension
            ) {
                const auto stride =
                    loop.strides[input.index][dimension];
                if (stride == 0) {
                    continue;
                }
                if (!first) {
                    expression += " + ";
                }
                first = false;
                expression += "index" + std::to_string(dimension);
                if (stride != 1) {
                    expression += " * " + std::to_string(stride);
                }
            }
            expression += first ? "0]" : "]";
            if (dtype == core::DType::float16) {
                return "ormi_half_value(" + expression + ")";
            }
            return expression;
        }

        const auto access = loop.vectorized
            ? loop.inputs[input.index]
            : Access::indexed;

        switch (access) {
            case Access::linear:
                expression += loop.span != 0 ? "_span[j]" : "[i]";
                break;
            case Access::scalar:
                expression += "_scalar";
                break;
            case Access::span:
                expression += "_span[j]";
                break;
            case Access::fixed:
                expression += "_fixed";
                break;
            case Access::indexed:
                expression += "[" + input_offset(input) + "]";
                break;
        }
        if (dtype == core::DType::float16) {
            return "ormi_half_value(" + expression + ")";
        }
        return expression;
    }

    std::string packed_input(
        const core::runtime::fusion::Input& input,
        core::DType dtype,
        const Loop& loop
    ) {
        if (loop.strided) {
            if (dtype != core::DType::float32) {
                ORMI_THROW(
                    OperationError,
                    "Packed strided fusion requires Float32 inputs."
                );
            }
            return packed_strided_input(input, loop);
        }

        const auto name = "input" + std::to_string(input.index);
        const bool float16 = dtype == core::DType::float16;
        switch (loop.inputs[input.index]) {
            case Access::linear:
                return float16
                    ? "ormi_half_load8(" + name +
                        (loop.span != 0 ? "_span + j)" : " + i)")
                    : "_mm256_loadu_ps(" + name +
                        (loop.span != 0 ? "_span + j)" : " + i)");
            case Access::scalar:
                return "_mm256_set1_ps(" + (float16
                    ? "ormi_half_value(" + name + "_scalar)"
                    : name + "_scalar") + ")";
            case Access::span:
                return float16
                    ? "ormi_half_load8(" + name + "_span + j)"
                    : "_mm256_loadu_ps(" + name + "_span + j)";
            case Access::fixed:
                return "_mm256_set1_ps(" + (float16
                    ? "ormi_half_value(" + name + "_fixed)"
                    : name + "_fixed") + ")";
            case Access::indexed:
                ORMI_THROW(
                    OperationError,
                    "Packed fusion requires planned input access."
                );
        }

        ORMI_THROW(
            OperationError,
            "Packed fusion received unknown input access."
        );
    }
}
