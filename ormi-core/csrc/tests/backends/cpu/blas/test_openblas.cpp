#include <gtest/gtest.h>

#include "backends/cpu/blas/openblas.h"

namespace {
    TEST(CPUOpenBLASTest, multipliesContiguousFloat32Matrices) {
        const float lhs[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        const float rhs[]{7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
        float out[4]{};

        ASSERT_TRUE(ormi::backends::cpu::blas::gemm(
            out,
            lhs,
            {2, 3, 3, 1},
            rhs,
            {3, 2, 2, 1}
        ));
        EXPECT_FLOAT_EQ(out[0], 58.0f);
        EXPECT_FLOAT_EQ(out[1], 64.0f);
        EXPECT_FLOAT_EQ(out[2], 139.0f);
        EXPECT_FLOAT_EQ(out[3], 154.0f);
    }

    TEST(CPUOpenBLASTest, multipliesTransposedFloat64View) {
        const double lhs[]{1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
        const double rhs[]{7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
        double out[4]{};

        ASSERT_TRUE(ormi::backends::cpu::blas::gemm(
            out,
            lhs,
            {2, 3, 1, 2},
            rhs,
            {3, 2, 2, 1}
        ));
        EXPECT_DOUBLE_EQ(out[0], 58.0);
        EXPECT_DOUBLE_EQ(out[1], 64.0);
        EXPECT_DOUBLE_EQ(out[2], 139.0);
        EXPECT_DOUBLE_EQ(out[3], 154.0);
    }

    TEST(CPUOpenBLASTest, declinesUnsupportedLayout) {
        const float lhs[]{1.0f, 2.0f};
        const float rhs[]{3.0f, 4.0f};
        float out[]{-1.0f};

        EXPECT_FALSE(ormi::backends::cpu::blas::gemm(
            out,
            lhs,
            {1, 2, 0, 0},
            rhs,
            {2, 1, 1, 1}
        ));
        EXPECT_FLOAT_EQ(out[0], -1.0f);
    }
}
