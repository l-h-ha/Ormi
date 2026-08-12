#include <limits>
#include <optional>
#include <string_view>

#if defined(ORMI_ENABLE_OPENBLAS)
#include <cblas.h>
#endif

#include "backends/cpu/blas/openblas.h"

namespace ormi::backends::cpu::blas {
#if defined(ORMI_ENABLE_OPENBLAS)
    namespace {
        struct Layout {
            CBLAS_TRANSPOSE transpose;
            blasint leading;
        };

        std::optional<Layout> layout(Matrix matrix) {
            const auto limit = static_cast<std::uint64_t>(std::numeric_limits<blasint>::max());

            if (
                matrix.column_stride == 1 &&
                matrix.row_stride >= static_cast<std::int64_t>(matrix.columns) &&
                static_cast<std::uint64_t>(matrix.row_stride) <= limit
            ) {
                return Layout{CblasNoTrans, static_cast<blasint>(matrix.row_stride)};
            }

            if (
                matrix.row_stride == 1 &&
                matrix.column_stride >= static_cast<std::int64_t>(matrix.rows) &&
                static_cast<std::uint64_t>(matrix.column_stride) <= limit
            ) {
                return Layout{CblasTrans, static_cast<blasint>(matrix.column_stride)};
            }

            return std::nullopt;
        }

        bool dimensions_fit(Matrix lhs, Matrix rhs) {
            const auto limit = static_cast<std::uint64_t>(std::numeric_limits<blasint>::max());
            return
                lhs.rows > 0 &&
                lhs.columns > 0 &&
                rhs.columns > 0 &&
                lhs.columns == rhs.rows &&
                lhs.rows <= limit &&
                lhs.columns <= limit &&
                rhs.columns <= limit;
        }

        template <typename T, typename Function>
        bool gemm_typed(
            T* out,
            const T* lhs,
            Matrix lhs_matrix,
            const T* rhs,
            Matrix rhs_matrix,
            Function function
        ) {
            if (!dimensions_fit(lhs_matrix, rhs_matrix)) {
                return false;
            }

            const auto lhs_layout = layout(lhs_matrix);
            const auto rhs_layout = layout(rhs_matrix);
            if (!lhs_layout || !rhs_layout) {
                return false;
            }

            function(
                CblasRowMajor,
                lhs_layout->transpose,
                rhs_layout->transpose,
                static_cast<blasint>(lhs_matrix.rows),
                static_cast<blasint>(rhs_matrix.columns),
                static_cast<blasint>(lhs_matrix.columns),
                static_cast<T>(1),
                lhs,
                lhs_layout->leading,
                rhs,
                rhs_layout->leading,
                static_cast<T>(0),
                out,
                static_cast<blasint>(rhs_matrix.columns)
            );
            return true;
        }
    }

    bool available() {
        return true;
    }

    bool generic() {
        static const bool result =
            std::string_view(openblas_get_corename()) == "generic";
        return result;
    }

    bool gemm(float* out, const float* lhs, Matrix lhs_matrix, const float* rhs, Matrix rhs_matrix) {
        return gemm_typed(out, lhs, lhs_matrix, rhs, rhs_matrix, cblas_sgemm);
    }

    bool gemm(double* out, const double* lhs, Matrix lhs_matrix, const double* rhs, Matrix rhs_matrix) {
        return gemm_typed(out, lhs, lhs_matrix, rhs, rhs_matrix, cblas_dgemm);
    }
#else
    bool available() {
        return false;
    }

    bool generic() {
        return false;
    }

    bool gemm(float*, const float*, Matrix, const float*, Matrix) {
        return false;
    }

    bool gemm(double*, const double*, Matrix, const double*, Matrix) {
        return false;
    }
#endif
}
