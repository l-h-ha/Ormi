#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include <ormi/core/error.h>
#include <ormi/core/tensor.h>

#include "core/runtime/backend.h"

namespace ormi::core::runtime {
    namespace {
        [[noreturn]] void unsupported(const Device& device, std::string_view operation) {
            ORMI_THROW(
                OperationError,
                "Backend '" + std::string(device.name()) + "' does not support " + std::string(operation) +
                    "."
            );
        }

        void unsupported_binary(ops::BinaryOp, Tensor& out, const Tensor&, const Tensor&) {
            unsupported(out.device(), "pointwise binary operations");
        }

        void unsupported_unary(ops::UnaryOp, Tensor& out, const Tensor&) {
            unsupported(out.device(), "pointwise unary operations");
        }

        void unsupported_cast(Tensor& out, const Tensor&) {
            unsupported(out.device(), "casts");
        }

        void unsupported_bernoulli(Tensor& out, const Bernoulli&) {
            unsupported(out.device(), "Bernoulli masks");
        }

        void unsupported_matmul(Tensor& out, const Tensor&, const Tensor&) {
            unsupported(out.device(), "matrix multiplication");
        }

        void unsupported_reduce(
            ops::ReduceOp,
            Tensor& out,
            const Tensor&,
            std::span<const std::int64_t>,
            bool
        ) {
            unsupported(out.device(), "reductions");
        }

        bool unsupported_finite(std::span<const Tensor* const> inputs) {
            if (inputs.empty() || inputs.front() == nullptr) {
                ORMI_THROW(OperationError, "The selected backend does not support finiteness queries.");
            }
            unsupported(inputs.front()->device(), "finiteness queries");
        }

        void unsupported_sgd(Tensor& out, const Tensor&, const Tensor&, double) {
            unsupported(out.device(), "SGD");
        }

        void
        unsupported_fractional(ops::FractionalOp, Tensor& out, const Tensor&, const Tensor*, const Caputo&) {
            unsupported(out.device(), "fractional differentiation");
        }

        void unsupported_index(Tensor& out, const Tensor&, const Tensor&, const Pick&) {
            unsupported(out.device(), "indexed selection");
        }

        void validate_memory(const Backend& backend, std::string_view name) {
            if (!backend.memory.allocator || !backend.memory.synchronize || !backend.memory.copy ||
                !backend.memory.copy_async || !backend.memory.fill || !backend.memory.fill_scalar) {
                ORMI_THROW(DeviceError, std::string(name) + " backend returned incomplete memory functions.");
            }
        }

        void validate_groups(const Backend& backend, std::string_view name) {
            const bool records =
                backend.recording.record || backend.recording.replay || backend.recording.release;
            const bool recording_complete =
                backend.recording.record && backend.recording.replay && backend.recording.release;
            if (records && !recording_complete) {
                ORMI_THROW(
                    DeviceError,
                    std::string(name) + " backend returned incomplete recording functions."
                );
            }

            if ((backend.fusion.supports || backend.fusion.prepare || backend.fusion.prepare_many) &&
                !backend.fusion.execute) {
                ORMI_THROW(
                    DeviceError,
                    std::string(name) + " backend returned fusion metadata without execution."
                );
            }

            if (backend.fractional.prepare && !backend.fractional.execute) {
                ORMI_THROW(
                    DeviceError,
                    std::string(name) + " backend returned fractional preparation without execution."
                );
            }
        }
    }

    Backend normalize(Backend backend, std::string_view name) {
        if (backend.abi != backend_abi) {
            ORMI_THROW(DeviceError, "Backend interface version does not match this ORMI core.");
        }

        validate_memory(backend, name);
        validate_groups(backend, name);

        if (!backend.ops.binary) {
            backend.ops.binary = unsupported_binary;
        }
        if (!backend.ops.unary) {
            backend.ops.unary = unsupported_unary;
        }
        if (!backend.ops.cast) {
            backend.ops.cast = unsupported_cast;
        }
        if (!backend.ops.matmul) {
            backend.ops.matmul = unsupported_matmul;
        }
        if (!backend.ops.reduce) {
            backend.ops.reduce = unsupported_reduce;
        }
        if (!backend.ops.all_finite) {
            backend.ops.all_finite = unsupported_finite;
        }
        if (!backend.ops.sgd) {
            backend.ops.sgd = unsupported_sgd;
        }
        if (!backend.ops.pick)
            backend.ops.pick = unsupported_index;
        if (!backend.ops.unpick)
            backend.ops.unpick = unsupported_index;
        if (!backend.ops.bernoulli) {
            backend.ops.bernoulli = unsupported_bernoulli;
        }
        if (!backend.fractional.execute) {
            backend.fractional.execute = unsupported_fractional;
        }

        return backend;
    }
}
