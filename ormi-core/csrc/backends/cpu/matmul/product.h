#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <ormi/core/float16.h>

#include "backends/cpu/blas/openblas.h"
#include "backends/cpu/matmul/portable.h"

#if defined(ORMI_HAS_AVX2_KERNELS)
#include "backends/cpu/matmul/avx2.h"
#include "platform/platform.h"
#endif

#if defined(_MSC_VER)
#define ORMI_GEMM_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define ORMI_GEMM_FORCE_INLINE inline __attribute__((always_inline))
#else
#define ORMI_GEMM_FORCE_INLINE inline
#endif

namespace ormi::backends::cpu::gemm {
    struct Problem {
        std::size_t rows;
        std::size_t inner;
        std::size_t columns;
        std::int64_t lhs_row_stride;
        std::int64_t lhs_column_stride;
        std::int64_t rhs_row_stride;
        std::int64_t rhs_column_stride;
        std::int64_t out_row_stride;
        std::int64_t out_column_stride;
    };

    namespace detail {
        constexpr std::size_t small_multiplications = 1ull << 20;
#if defined(ORMI_HAS_AVX2_KERNELS)
        constexpr std::size_t avx2_multiplications = 1ull << 12;
#endif

        inline bool small(const Problem& problem) {
            return
                problem.columns >= 4 &&
                problem.rows <= small_multiplications / problem.inner &&
                problem.rows * problem.inner <=
                    small_multiplications / problem.columns;
        }

        template <typename T>
        ORMI_GEMM_FORCE_INLINE void run(
            T* out,
            const T* lhs,
            const T* rhs,
            const Problem& problem
        ) {
#if defined(ORMI_HAS_AVX2_KERNELS)
            static const bool has_fma = platform::supports_fma();
            constexpr bool float32 = std::is_same_v<T, float>;
            const bool use_avx2 = float32 && has_fma;
#endif

            if (problem.columns == 1) {
                const bool column_lhs =
                    problem.lhs_row_stride == 1 &&
                    problem.out_row_stride == 1;
                if (column_lhs) {
                    portable::column_lhs_gemv(
                        out,
                        lhs,
                        rhs,
                        problem.rows,
                        problem.inner,
                        problem.lhs_column_stride,
                        problem.rhs_row_stride
                    );
                    return;
                }
#if defined(ORMI_HAS_AVX2_KERNELS)
                const bool enough_work =
                    problem.rows > avx2_multiplications / problem.inner ||
                    problem.rows * problem.inner >= avx2_multiplications;
                if constexpr (float32) {
                    if (
                        use_avx2 &&
                        enough_work &&
                        problem.lhs_column_stride == 1 &&
                        problem.rhs_row_stride == 1 &&
                        problem.out_row_stride == 1
                    ) {
                        avx2::row_gemv(
                            out,
                            lhs,
                            rhs,
                            problem.rows,
                            problem.inner,
                            problem.lhs_row_stride,
                            problem.out_row_stride
                        );
                        return;
                    }
                }
#endif
                portable::row_gemv(
                    out,
                    lhs,
                    rhs,
                    problem.rows,
                    problem.inner,
                    problem.lhs_row_stride,
                    problem.lhs_column_stride,
                    problem.rhs_row_stride,
                    problem.out_row_stride
                );
                return;
            }

            bool use_internal = small(problem);
#if defined(ORMI_HAS_AVX2_KERNELS)
            if constexpr (float32) {
                if (
                    use_avx2 &&
                    problem.columns >= 4 &&
                    (!blas::available() || blas::generic())
                ) {
                    use_internal = true;
                }
            }
#endif

            if (
                use_internal &&
                problem.rhs_column_stride == 1 &&
                problem.out_column_stride == 1
            ) {
                const bool column_lhs =
                    problem.lhs_row_stride == 1 &&
                    problem.out_row_stride ==
                        static_cast<std::int64_t>(problem.columns);

#if defined(ORMI_HAS_AVX2_KERNELS)
                if constexpr (float32) {
                    const auto multiplications =
                        problem.rows * problem.inner * problem.columns;
                    if (
                        use_avx2 &&
                        problem.columns >= 8 &&
                        multiplications >= avx2_multiplications
                    ) {
                        if (column_lhs) {
                            avx2::parallel_column_lhs_gemm(
                                out,
                                lhs,
                                rhs,
                                problem.rows,
                                problem.inner,
                                problem.columns,
                                problem.lhs_column_stride,
                                problem.rhs_row_stride,
                                multiplications
                            );
                        } else {
                            avx2::parallel_row_gemm(
                                out,
                                lhs,
                                rhs,
                                problem.rows,
                                problem.inner,
                                problem.columns,
                                problem.lhs_row_stride,
                                problem.lhs_column_stride,
                                problem.rhs_row_stride,
                                problem.out_row_stride,
                                multiplications
                            );
                        }
                        return;
                    }
                }
#endif

                if (column_lhs) {
                    portable::column_lhs_gemm(
                        out,
                        lhs,
                        rhs,
                        problem.rows,
                        problem.inner,
                        problem.columns,
                        problem.lhs_column_stride,
                        problem.rhs_row_stride
                    );
                    return;
                }

                portable::row_gemm(
                    out,
                    lhs,
                    rhs,
                    problem.rows,
                    problem.inner,
                    problem.columns,
                    problem.lhs_row_stride,
                    problem.lhs_column_stride,
                    problem.rhs_row_stride,
                    problem.out_row_stride
                );
                return;
            }

            if constexpr (
                std::is_same_v<T, float> ||
                std::is_same_v<T, double>
            ) {
                const auto lhs_matrix = blas::Matrix{
                    problem.rows,
                    problem.inner,
                    problem.lhs_row_stride,
                    problem.lhs_column_stride
                };
                const auto rhs_matrix = blas::Matrix{
                    problem.inner,
                    problem.columns,
                    problem.rhs_row_stride,
                    problem.rhs_column_stride
                };
                if (blas::gemm(
                    out,
                    lhs,
                    lhs_matrix,
                    rhs,
                    rhs_matrix
                )) {
                    return;
                }
            }

            portable::strided(
                out,
                lhs,
                rhs,
                problem.rows,
                problem.inner,
                problem.columns,
                problem.lhs_row_stride,
                problem.lhs_column_stride,
                problem.rhs_row_stride,
                problem.rhs_column_stride,
                problem.out_row_stride,
                problem.out_column_stride
            );
        }
    }

    ORMI_GEMM_FORCE_INLINE void run(
        double* out,
        const double* lhs,
        const double* rhs,
        const Problem& problem
    ) {
        detail::run(out, lhs, rhs, problem);
    }

    ORMI_GEMM_FORCE_INLINE void run(
        float* out,
        const float* lhs,
        const float* rhs,
        const Problem& problem
    ) {
        detail::run(out, lhs, rhs, problem);
    }

    ORMI_GEMM_FORCE_INLINE void run(
        core::Float16* out,
        const core::Float16* lhs,
        const core::Float16* rhs,
        const Problem& problem
    ) {
        portable::strided(
            out,
            lhs,
            rhs,
            problem.rows,
            problem.inner,
            problem.columns,
            problem.lhs_row_stride,
            problem.lhs_column_stride,
            problem.rhs_row_stride,
            problem.rhs_column_stride,
            problem.out_row_stride,
            problem.out_column_stride
        );
    }

    ORMI_GEMM_FORCE_INLINE void run(
        std::int32_t* out,
        const std::int32_t* lhs,
        const std::int32_t* rhs,
        const Problem& problem
    ) {
        detail::run(out, lhs, rhs, problem);
    }
}

#undef ORMI_GEMM_FORCE_INLINE
