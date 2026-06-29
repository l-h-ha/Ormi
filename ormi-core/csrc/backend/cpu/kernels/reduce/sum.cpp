#include <cstdint>
#include <memory>
#include <vector>

#include <half.hpp>

#include "backend/graph_breakers.h"
#include "core/allocator.h"
#include "core/dtype.h"
#include "core/error.h"
#include "core/buffer.h"
#include "core/node.h"
#include "core/shape.h"
#include "sum.h"

namespace ormi::cpu {
    namespace {
        size_t numel_from_shape(const std::vector<int>& shape) {
            if (shape.empty()) {
                return 1;
            }

            size_t n = 1;
            for (int dim : shape) {
                n *= static_cast<size_t>(dim);
            }

            return n;
        }

        std::vector<bool> make_reduced_mask(const std::vector<int>& axes, int ndim) {
            std::vector<bool> reduced(static_cast<size_t>(ndim), false);

            if (axes.empty()) {
                for (int d = 0; d < ndim; ++d) {
                    reduced[static_cast<size_t>(d)] = true;
                }
                return reduced;
            }

            for (int axis : axes) {
                int normalized = axis;
                if (normalized < 0) {
                    normalized += ndim;
                }

                if (normalized < 0 || normalized >= ndim) {
                    ORMI_THROW(ShapeError, "SUM axis out of bounds.");
                }
                reduced[static_cast<size_t>(normalized)] = true;
            }

            return reduced;
        }

        template <typename T>
        void sum_impl(
            const std::shared_ptr<Buffer>& in_buf,
            const TensorView& in_view,
            const std::vector<int>& axes,
            const TensorView& out_view,
            bool keepdims,
            const std::shared_ptr<Buffer>& out_buffer
        ) {
            const int ndim = static_cast<int>(in_view.shape.size());
            const size_t logical_elements = numel_from_shape(in_view.shape);

            const auto reduced = make_reduced_mask(axes, ndim);

            const auto* in_ptr = static_cast<const T*>(in_buf->ptr);
            auto* out_ptr = static_cast<T*>(out_buffer->ptr);

            std::vector<int> coords(static_cast<size_t>(ndim), 0);

            for (size_t logical_i = 0; logical_i < logical_elements; ++logical_i) {
                size_t remaining = logical_i;

                for (int d = ndim - 1; d >= 0; --d) {
                    const int dim = in_view.shape[static_cast<size_t>(d)];

                    coords[static_cast<size_t>(d)] =
                        static_cast<int>(remaining % static_cast<size_t>(dim));

                    remaining /= static_cast<size_t>(dim);
                }

                size_t in_offset = 0;
                size_t out_flat = 0;
                int out_d = 0;

                for (int d = 0; d < ndim; ++d) {
                    const int coord = coords[static_cast<size_t>(d)];

                    in_offset += static_cast<size_t>(
                        coord * in_view.strides[static_cast<size_t>(d)]
                    );

                    if (!reduced[static_cast<size_t>(d)]) {
                        out_flat += static_cast<size_t>(
                            coord * out_view.strides[static_cast<size_t>(out_d)]
                        );
                        ++out_d;
                    } else if (keepdims) {
                        ++out_d;
                    }
                }

                out_ptr[out_flat] += in_ptr[in_offset];
            }
        }

        template <>
        void sum_impl<half_float::half>(
            const std::shared_ptr<Buffer>& in_buf,
            const TensorView& in_view,
            const std::vector<int>& axes,
            const TensorView& out_view,
            bool keepdims,
            const std::shared_ptr<Buffer>& out_buffer
        ) {
            const int ndim = static_cast<int>(in_view.shape.size());
            const size_t logical_elements = numel_from_shape(in_view.shape);

            const auto reduced = make_reduced_mask(axes, ndim);

            const auto* in_ptr = static_cast<const half_float::half*>(in_buf->ptr);
            auto* out_ptr = static_cast<half_float::half*>(out_buffer->ptr);

            std::vector<int> coords(static_cast<size_t>(ndim), 0);

            for (size_t logical_i = 0; logical_i < logical_elements; ++logical_i) {
                size_t remaining = logical_i;

                for (int d = ndim - 1; d >= 0; --d) {
                    const int dim = in_view.shape[static_cast<size_t>(d)];

                    coords[static_cast<size_t>(d)] =
                        static_cast<int>(remaining % static_cast<size_t>(dim));

                    remaining /= static_cast<size_t>(dim);
                }

                size_t in_offset = 0;
                size_t out_flat = 0;
                int out_d = 0;

                for (int d = 0; d < ndim; ++d) {
                    const int coord = coords[static_cast<size_t>(d)];

                    in_offset += static_cast<size_t>(
                        coord * in_view.strides[static_cast<size_t>(d)]
                    );

                    if (!reduced[static_cast<size_t>(d)]) {
                        out_flat += static_cast<size_t>(
                            coord * out_view.strides[static_cast<size_t>(out_d)]
                        );
                        ++out_d;
                    } else if (keepdims) {
                        ++out_d;
                    }
                }

                const float acc =
                    static_cast<float>(out_ptr[out_flat]) +
                    static_cast<float>(in_ptr[in_offset]);

                out_ptr[out_flat] = half_float::half(acc);
            }
        }

    }

    std::shared_ptr<Buffer> sum(
        const std::shared_ptr<Buffer>& in_buf,
        const TensorView& in_view,
        const std::vector<int>& axes,
        const TensorView& out_view,
        const bool keepdims,
        std::shared_ptr<Buffer> out_buffer
    ) {
        if (!in_buf || !in_buf->ptr) {
            ORMI_THROW(MemoryError, "CPU SUM received null input buffer.");
        }

        if (!out_buffer) {
            out_buffer = zeros(out_view.shape, in_buf->device, in_buf->dtype);
        } else {
            get_allocator(out_buffer->device)->memset(
                out_buffer->ptr,
                0,
                out_buffer->bytes
            );
        }

        if (in_buf->dtype != out_buffer->dtype) {
            ORMI_THROW(DTypeError, "CPU SUM input/output dtype mismatch.");
        }

        switch (in_buf->dtype) {
            case DType::Float32:
                sum_impl<float>(
                    in_buf,
                    in_view,
                    axes,
                    out_view,
                    keepdims,
                    out_buffer
                );
                break;

            case DType::Float16:
                static_assert(
                    sizeof(half_float::half) == 2,
                    "half_float::half must be 16 bits."
                );

                sum_impl<half_float::half>(
                    in_buf,
                    in_view,
                    axes,
                    out_view,
                    keepdims,
                    out_buffer
                );
                break;

            case DType::Int32:
                sum_impl<int32_t>(
                    in_buf,
                    in_view,
                    axes,
                    out_view,
                    keepdims,
                    out_buffer
                );
                break;

            case DType::Bool:
                ORMI_THROW(
                    DTypeError,
                    "CPU SUM does not currently support Bool. "
                );

            default:
                ORMI_THROW(DTypeError, "Unrecognized dtype.");
        }

        return out_buffer;
    }

    void sum_handler(const std::shared_ptr<Node>& root) {
        const auto& attrs = std::get<ReduceAttrs>(root->attrs);

        if (!root->data) {
            root->data = zeros(root->view.shape, root->device, root->dtype);
        }

        root->data = cpu::sum(
            root->parents[0]->data,
            root->parents[0]->view,
            attrs.axes,
            root->view,
            attrs.keepdims,
            root->data
        );
    }

    static RegisterGraphBreaker reg_cpu_sum(
        DeviceEnum::CPU,
        ops::OpEnum::SUM,
        sum_handler
    );

}