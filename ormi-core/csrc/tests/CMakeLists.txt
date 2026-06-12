cmake_minimum_required(VERSION 3.18)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

target_include_directories(ormi_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(ormi_backend_tests
  test_utils.h
  core/test_memory.cpp
  core/test_ops.cpp
  core/test_shape.cpp
  core/test_node.cpp
  core/test_autograd.cpp
  core/test_allocator.cpp
  core/ops/blas/test_matmul.cpp
  core/ops/reduce/test_reduce.cpp
  core/ops/view/test_expand.cpp
  core/ops/view/test_reshape.cpp
  core/ops/view/test_transpose.cpp
  core/ops/pointwise/test_add.cpp
  core/ops/pointwise/test_cos.cpp
  core/ops/pointwise/test_div.cpp
  core/ops/pointwise/test_exp.cpp
  core/ops/pointwise/test_mul.cpp
  core/ops/pointwise/test_neg.cpp
  core/ops/pointwise/test_sin.cpp
  core/ops/pointwise/test_sub.cpp
  backend/test_runtime.cpp
  backend/test_graph_breakers.cpp
  backend/cpu/test_compiler.cpp
  backend/cpu/test_jit.cpp
  backend/cpu/kernels/blas/test_matmul.cpp
  backend/cpu/kernels/reduce/test_sum.cpp
  backend/cuda/test_compiler.cpp
  backend/cuda/test_jit.cpp
  backend/cuda/kernels/blas/test_matmul.cpp
  backend/cuda/kernels/reduce/test_sum.cpp
  core/ops/pointwise/test_log.cpp
  core/ops/pointwise/test_pow.cpp)
target_compile_features(ormi_backend_tests PRIVATE cxx_std_17)

target_include_directories(ormi_backend_tests PRIVATE ${CMAKE_SOURCE_DIR}/csrc)

target_link_libraries(ormi_backend_tests PRIVATE 
    ormi_engine 
    gtest
    gtest_main
)

include(GoogleTest)
gtest_discover_tests(ormi_backend_tests)