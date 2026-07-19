if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)
find_package(Threads REQUIRED)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(ORMI_RUNTIME_TEST_SRCS
    core/test_autocast.cpp
    core/test_dtype.cpp
    core/test_float16.cpp
    core/test_shape.cpp
    core/test_tensor.cpp
    core/test_autograd.cpp
    core/autograd/test_linalg.cpp
    core/autograd/test_reduction.cpp
    core/autograd/test_view.cpp
    core/ops/test_cast.cpp
    core/ops/test_creation.cpp
    core/ops/test_finite.cpp
    core/ops/test_linalg.cpp
    core/ops/test_pointwise.cpp
    core/ops/test_reduction.cpp
    core/ops/test_view.cpp
    core/optim/test_grad_scaler.cpp
    core/optim/test_sgd.cpp
    memory/test_allocator.cpp
    memory/test_buffer.cpp
    memory/test_ops.cpp
    platform/test_process.cpp
    runtime/test_backends.cpp
    runtime/test_device.cpp
    runtime/test_finite.cpp
    runtime/test_runtime_dispatch.cpp
    runtime/fusion/test_expr.cpp
)

set(ORMI_BACKEND_TEST_SRCS
    backends/cpu/test_cpu_backend.cpp
    backends/cpu/jit/test_cache.cpp
    backends/cpu/jit/test_codegen.cpp
    backends/cpu/jit/test_compiler.cpp
    backends/cpu/jit/test_kernel.cpp
)

if(ORMI_ENABLE_OPENBLAS)
    list(APPEND ORMI_BACKEND_TEST_SRCS
        backends/cpu/blas/test_openblas.cpp
    )
endif()

add_executable(ormi_runtime_tests
    ${ORMI_RUNTIME_TEST_SRCS}
    ${ORMI_BACKEND_TEST_SRCS}
)

add_executable(ormi_jit_cache_helper
    helpers/jit_cache.cpp
)
target_compile_features(ormi_jit_cache_helper PRIVATE cxx_std_20)
target_link_libraries(ormi_jit_cache_helper PRIVATE ormi::core)
ormi_stage_runtime(ormi_jit_cache_helper)

add_executable(ormi_process_helper
    helpers/process.cpp
)
target_compile_features(ormi_process_helper PRIVATE cxx_std_20)

add_dependencies(
    ormi_runtime_tests
    ormi_jit_cache_helper
    ormi_process_helper
)
target_compile_definitions(ormi_runtime_tests PRIVATE
    ORMI_JIT_CACHE_HELPER="$<TARGET_FILE:ormi_jit_cache_helper>"
    ORMI_PROCESS_HELPER="$<TARGET_FILE:ormi_process_helper>"
)

target_compile_features(ormi_runtime_tests PRIVATE cxx_std_20)
target_include_directories(ormi_runtime_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_link_libraries(ormi_runtime_tests PRIVATE
    ormi::core
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
)

if(ORMI_ENABLE_CUDA)
    target_compile_definitions(ormi_runtime_tests PRIVATE ORMI_ENABLE_CUDA)
endif()
ormi_stage_runtime(ormi_runtime_tests)

include(GoogleTest)
gtest_discover_tests(
    ormi_runtime_tests
    PROPERTIES ENVIRONMENT
        "ORMI_CPU_JIT_CACHE_DIR=${CMAKE_CURRENT_BINARY_DIR}/jit-cache"
)

add_executable(ormi_backend_loading_tests
    runtime/test_backend_loading.cpp
)
target_compile_features(ormi_backend_loading_tests PRIVATE cxx_std_20)
target_link_libraries(ormi_backend_loading_tests PRIVATE
    ormi::core
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
)
if(ORMI_ENABLE_CUDA)
    target_compile_definitions(ormi_backend_loading_tests PRIVATE ORMI_ENABLE_CUDA)
endif()
ormi_stage_runtime(ormi_backend_loading_tests)
gtest_discover_tests(ormi_backend_loading_tests)

if(ORMI_ENABLE_CUDA)
    add_executable(ormi_cuda_backend_tests
        ../backends/cuda/fusion/codegen.cpp
        backends/cuda/test_cuda_allocator.cpp
        backends/cuda/test_cuda_backend.cpp
        backends/cuda/test_cuda_cast.cpp
        backends/cuda/test_cuda_creation.cpp
        backends/cuda/test_cuda_finite.cpp
        backends/cuda/test_cuda_fusion.cpp
        backends/cuda/test_cuda_linalg.cpp
        backends/cuda/test_cuda_memory.cpp
        backends/cuda/test_cuda_optimizer.cpp
        backends/cuda/test_cuda_pointwise.cpp
        backends/cuda/test_cuda_reduction.cpp
    )
    target_compile_features(ormi_cuda_backend_tests PRIVATE cxx_std_20)
    target_include_directories(ormi_cuda_backend_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
    )
    target_link_libraries(ormi_cuda_backend_tests PRIVATE
        ormi::core
        ormi::cuda
        CUDA::cudart
        CUDA::cuda_driver
        GTest::gtest
        GTest::gtest_main
    )
    ormi_stage_runtime(ormi_cuda_backend_tests)
    gtest_discover_tests(ormi_cuda_backend_tests)

    add_executable(ormi_cuda_state_tests
        backends/cuda/test_cuda_state.cpp
        ../backends/cuda/state.cu
        ../backends/cuda/check.cu
    )
    add_dependencies(ormi_cuda_state_tests ormi_cuda_backend)
    set_target_properties(ormi_cuda_state_tests PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )
    target_compile_features(ormi_cuda_state_tests PRIVATE cxx_std_20)
    target_include_directories(ormi_cuda_state_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
    )
    target_link_libraries(ormi_cuda_state_tests PRIVATE
        ormi::core
        CUDA::cudart
        CUDA::cublas
        GTest::gtest
        GTest::gtest_main
    )
    ormi_stage_runtime(ormi_cuda_state_tests)
    gtest_discover_tests(ormi_cuda_state_tests)
endif()
