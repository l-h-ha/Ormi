#include <cstddef>
#include <cstdint>

#include "backends/cpu/executor.h"
#include "backends/cpu/matmul/avx2.h"

namespace ormi::backends::cpu::avx2 {
    namespace {
        struct RowGEMM {
            float* out;
            const float* lhs;
            const float* rhs;
            std::size_t inner;
            std::size_t columns;
            std::int64_t lhs_row_stride;
            std::int64_t lhs_column_stride;
            std::int64_t rhs_row_stride;
            std::int64_t out_row_stride;
        };

        void run_row_gemm(
            void* opaque,
            std::size_t,
            std::size_t begin,
            std::size_t end
        ) {
            auto& gemm = *static_cast<RowGEMM*>(opaque);
            row_gemm(
                gemm.out + begin *
                    static_cast<std::size_t>(gemm.out_row_stride),
                gemm.lhs + begin *
                    static_cast<std::size_t>(gemm.lhs_row_stride),
                gemm.rhs,
                end - begin,
                gemm.inner,
                gemm.columns,
                gemm.lhs_row_stride,
                gemm.lhs_column_stride,
                gemm.rhs_row_stride,
                gemm.out_row_stride
            );
        }

        struct ColumnLhsGEMM {
            float* out;
            const float* lhs;
            const float* rhs;
            std::size_t inner;
            std::size_t columns;
            std::int64_t lhs_column_stride;
            std::int64_t rhs_row_stride;
        };

        void run_column_lhs_gemm(
            void* opaque,
            std::size_t,
            std::size_t begin,
            std::size_t end
        ) {
            auto& gemm = *static_cast<ColumnLhsGEMM*>(opaque);
            column_lhs_gemm(
                gemm.out + begin * gemm.columns,
                gemm.lhs + begin,
                gemm.rhs,
                end - begin,
                gemm.inner,
                gemm.columns,
                gemm.lhs_column_stride,
                gemm.rhs_row_stride
            );
        }
    }

    void parallel_row_gemm(
        float* out,
        const float* lhs,
        const float* rhs,
        std::size_t rows,
        std::size_t inner,
        std::size_t columns,
        std::int64_t lhs_row_stride,
        std::int64_t lhs_column_stride,
        std::int64_t rhs_row_stride,
        std::int64_t out_row_stride,
        std::size_t work
    ) {
        constexpr auto row_tile = row_gemm_tile;
        if (
            rows < row_tile ||
            executor::slots(rows, row_tile, work) == 1
        ) {
            row_gemm(
                out,
                lhs,
                rhs,
                rows,
                inner,
                columns,
                lhs_row_stride,
                lhs_column_stride,
                rhs_row_stride,
                out_row_stride
            );
            return;
        }

        RowGEMM gemm{
            out,
            lhs,
            rhs,
            inner,
            columns,
            lhs_row_stride,
            lhs_column_stride,
            rhs_row_stride,
            out_row_stride
        };
        executor::run(rows, row_tile, work, run_row_gemm, &gemm);
    }

    void parallel_column_lhs_gemm(
        float* out,
        const float* lhs,
        const float* rhs,
        std::size_t rows,
        std::size_t inner,
        std::size_t columns,
        std::int64_t lhs_column_stride,
        std::int64_t rhs_row_stride,
        std::size_t work
    ) {
        constexpr auto row_tile = column_lhs_tile;
        if (
            rows < row_tile ||
            executor::slots(rows, row_tile, work) == 1
        ) {
            column_lhs_gemm(
                out,
                lhs,
                rhs,
                rows,
                inner,
                columns,
                lhs_column_stride,
                rhs_row_stride
            );
            return;
        }

        ColumnLhsGEMM gemm{
            out,
            lhs,
            rhs,
            inner,
            columns,
            lhs_column_stride,
            rhs_row_stride
        };
        executor::run(rows, row_tile, work, run_column_lhs_gemm, &gemm);
    }
}
