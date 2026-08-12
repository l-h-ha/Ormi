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
            std::size_t rows;
            std::size_t inner;
            std::size_t columns;
            std::int64_t lhs_row_stride;
            std::int64_t lhs_column_stride;
            std::int64_t rhs_row_stride;
            std::int64_t out_row_stride;
            std::size_t split;
        };

        void run_row_gemm(
            void* opaque,
            std::size_t,
            std::size_t begin,
            std::size_t end
        ) {
            auto& gemm = *static_cast<RowGEMM*>(opaque);
            for (std::size_t partition = begin; partition < end; ++partition) {
                const auto row_begin = partition == 0 ? 0 : gemm.split;
                const auto row_end = partition == 0 ? gemm.split : gemm.rows;
                row_gemm(
                    gemm.out + row_begin *
                        static_cast<std::size_t>(gemm.out_row_stride),
                    gemm.lhs + row_begin *
                        static_cast<std::size_t>(gemm.lhs_row_stride),
                    gemm.rhs,
                    row_end - row_begin,
                    gemm.inner,
                    gemm.columns,
                    gemm.lhs_row_stride,
                    gemm.lhs_column_stride,
                    gemm.rhs_row_stride,
                    gemm.out_row_stride
                );
            }
        }

        struct ColumnLhsGEMM {
            float* out;
            const float* lhs;
            const float* rhs;
            std::size_t rows;
            std::size_t inner;
            std::size_t columns;
            std::int64_t lhs_column_stride;
            std::int64_t rhs_row_stride;
            std::size_t split;
        };

        void run_column_lhs_gemm(
            void* opaque,
            std::size_t,
            std::size_t begin,
            std::size_t end
        ) {
            auto& gemm = *static_cast<ColumnLhsGEMM*>(opaque);
            for (std::size_t partition = begin; partition < end; ++partition) {
                const auto row_begin = partition == 0 ? 0 : gemm.split;
                const auto row_end = partition == 0 ? gemm.split : gemm.rows;
                column_lhs_gemm(
                    gemm.out + row_begin * gemm.columns,
                    gemm.lhs + row_begin,
                    gemm.rhs,
                    row_end - row_begin,
                    gemm.inner,
                    gemm.columns,
                    gemm.lhs_column_stride,
                    gemm.rhs_row_stride
                );
            }
        }

        std::size_t split(std::size_t rows, std::size_t tile) {
            const auto half = rows / 2;
            return half - half % tile;
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
        const auto row_split = split(rows, 2);
        if (
            row_split == 0 ||
            executor::slots(2, 1, work) == 1
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
            rows,
            inner,
            columns,
            lhs_row_stride,
            lhs_column_stride,
            rhs_row_stride,
            out_row_stride,
            row_split
        };
        executor::run(2, 1, work, run_row_gemm, &gemm);
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
        const auto row_split = split(rows, 4);
        if (
            row_split == 0 ||
            executor::slots(2, 1, work) == 1
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
            rows,
            inner,
            columns,
            lhs_column_stride,
            rhs_row_stride,
            row_split
        };
        executor::run(2, 1, work, run_column_lhs_gemm, &gemm);
    }
}
