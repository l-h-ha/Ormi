if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)
find_package(Threads REQUIRED)

add_test(
    NAME BackendCatalogTest
    COMMAND ${CMAKE_COMMAND}
        -DORMI_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/..
        -DORMI_TEST_DIR=${CMAKE_CURRENT_BINARY_DIR}/backend-catalog
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_backend_catalog.cmake
)
add_test(
    NAME PlatformRegistrationTest
    COMMAND ${CMAKE_COMMAND}
        -DORMI_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/..
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_platform.cmake
)
add_test(
    NAME PlatformUnsupportedTest
    COMMAND ${CMAKE_COMMAND}
        -DORMI_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/..
        -DORMI_TEST_UNSUPPORTED=ON
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_platform.cmake
)
set_tests_properties(PlatformUnsupportedTest PROPERTIES WILL_FAIL TRUE)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(ormi_api_test
    public/api.cpp
)
target_compile_features(ormi_api_test PRIVATE cxx_std_20)
target_link_libraries(ormi_api_test PRIVATE ormi::core)
ormi_stage_runtime(ormi_api_test)
add_test(NAME ApiTest COMMAND ormi_api_test)

set(ORMI_RUNTIME_TEST_SRCS
    core/test_autocast.cpp
    core/test_dtype.cpp
    core/test_float16.cpp
    core/test_shape.cpp
    core/test_tensor.cpp
    core/test_autograd.cpp
    core/test_training.cpp
    core/autograd/test_fractional.cpp
    core/autograd/test_linalg.cpp
    core/autograd/test_reduction.cpp
    core/autograd/test_view.cpp
    core/ops/test_cast.cpp
    core/ops/test_creation.cpp
    core/ops/test_finite.cpp
    core/ops/test_fractional.cpp
    core/ops/test_fractional_accuracy.cpp
    core/ops/test_linalg.cpp
    core/ops/test_pointwise.cpp
    core/ops/test_reduction.cpp
    core/ops/test_view.cpp
    core/optim/test_grad_scaler.cpp
    core/optim/test_sgd.cpp
    memory/test_allocator.cpp
    memory/test_buffer.cpp
    memory/test_ops.cpp
    platform/test_cpu.cpp
    platform/test_process.cpp
    runtime/test_backends.cpp
    runtime/test_device.cpp
    runtime/test_finite.cpp
    runtime/test_schedule.cpp
    core/test_plan.cpp
    runtime/test_runtime_dispatch.cpp
    runtime/fusion/test_expr.cpp
    runtime/fusion/test_group.cpp
    runtime/fusion/test_key.cpp
)

set(ORMI_BACKEND_TEST_SRCS
    backends/cpu/test_executor.cpp
    backends/cpu/test_cpu_backend.cpp
    backends/cpu/jit/test_accuracy.cpp
    backends/cpu/jit/test_cache.cpp
    backends/cpu/jit/test_codegen.cpp
    backends/cpu/jit/test_compile.cpp
    backends/cpu/jit/test_compiler.cpp
    backends/cpu/jit/test_kernel.cpp
    backends/cpu/jit/test_loop.cpp
    backends/cpu/jit/test_target.cpp
)

if(ORMI_ENABLE_OPENBLAS)
    list(APPEND ORMI_BACKEND_TEST_SRCS
        backends/cpu/blas/test_openblas.cpp
    )
endif()

add_executable(ormi_runtime_tests
    helpers/main.cpp
    ${ORMI_RUNTIME_TEST_SRCS}
    ${ORMI_BACKEND_TEST_SRCS}
)

add_executable(ormi_jit_cache_helper
    helpers/jit_cache.cpp
)
target_compile_features(ormi_jit_cache_helper PRIVATE cxx_std_20)
target_link_libraries(ormi_jit_cache_helper PRIVATE ormi::core)
ormi_stage_runtime(ormi_jit_cache_helper)

add_executable(ormi_jit_compiler_helper
    helpers/jit_compiler.cpp
)
target_compile_features(ormi_jit_compiler_helper PRIVATE cxx_std_20)
target_include_directories(ormi_jit_compiler_helper PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_link_libraries(ormi_jit_compiler_helper PRIVATE ormi::core)
if(MSVC)
    target_compile_options(ormi_jit_compiler_helper PRIVATE /EHsc)
endif()
ormi_stage_runtime(ormi_jit_compiler_helper)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(ORMI_TEST_CXX_ID gcc)
elseif(
    CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND
    CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"
)
    set(ORMI_TEST_CXX_ID clang-cl)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(ORMI_TEST_CXX_ID clang)
elseif(MSVC)
    set(ORMI_TEST_CXX_ID msvc)
else()
    message(FATAL_ERROR "No CPU JIT compiler id for ${CMAKE_CXX_COMPILER_ID}.")
endif()

set(ORMI_TEST_JIT_COMPILER_CACHE
    "${CMAKE_CURRENT_BINARY_DIR}/jit-compiler-cache"
)
add_test(
    NAME CPUJITCompilerMatrixTest
    COMMAND ormi_jit_compiler_helper
        verify
        "${ORMI_TEST_JIT_COMPILER_CACHE}"
        "${CMAKE_CXX_COMPILER}"
        "${ORMI_TEST_CXX_ID}"
)
add_test(
    NAME CPUJITMissingCompilerTest
    COMMAND ormi_jit_compiler_helper
        missing
        "${ORMI_TEST_JIT_COMPILER_CACHE}"
)

add_executable(ormi_process_helper
    helpers/process.cpp
)
target_compile_features(ormi_process_helper PRIVATE cxx_std_20)

add_dependencies(
    ormi_runtime_tests
    ormi_jit_cache_helper
    ormi_jit_compiler_helper
    ormi_process_helper
)
target_compile_definitions(ormi_runtime_tests PRIVATE
    ORMI_JIT_CACHE_HELPER="$<TARGET_FILE:ormi_jit_cache_helper>"
    ORMI_PROCESS_HELPER="$<TARGET_FILE:ormi_process_helper>"
)

target_compile_features(ormi_runtime_tests PRIVATE cxx_std_20)
target_include_directories(ormi_runtime_tests PRIVATE
    ${ORMI_GENERATED_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_link_libraries(ormi_runtime_tests PRIVATE
    ormi::core
    GTest::gtest
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
    helpers/main.cpp
    runtime/test_backend_loading.cpp
)
target_compile_features(ormi_backend_loading_tests PRIVATE cxx_std_20)
target_include_directories(ormi_backend_loading_tests PRIVATE
    ${ORMI_GENERATED_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_link_libraries(ormi_backend_loading_tests PRIVATE
    ormi::core
    GTest::gtest
    Threads::Threads
)
if(ORMI_ENABLE_CUDA)
    target_compile_definitions(ormi_backend_loading_tests PRIVATE ORMI_ENABLE_CUDA)
endif()
ormi_stage_runtime(ormi_backend_loading_tests)
gtest_discover_tests(ormi_backend_loading_tests)

if(ORMI_ENABLE_CUDA)
    add_executable(ormi_cuda_cache_helper
        helpers/cuda_cache.cpp
        ../backends/cuda/fusion/cache.cpp
    )
    target_compile_features(
        ormi_cuda_cache_helper
        PRIVATE cxx_std_20
    )
    target_include_directories(
        ormi_cuda_cache_helper
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..
    )
    target_link_libraries(
        ormi_cuda_cache_helper
        PRIVATE ormi::core
    )
    ormi_stage_runtime(ormi_cuda_cache_helper)

    add_executable(ormi_cuda_backend_tests
        helpers/main.cpp
        ../backends/cuda/fusion/cache.cpp
        ../backends/cuda/fusion/codegen.cpp
        ../backends/cuda/fusion/order.cpp
        ../backends/cuda/fusion/tile.cpp
        backends/cuda/test_cuda_allocator.cpp
        backends/cuda/test_cuda_backend.cpp
        backends/cuda/test_cuda_cast.cpp
        backends/cuda/test_cuda_creation.cpp
        backends/cuda/test_cuda_fusion_cache.cpp
        backends/cuda/test_cuda_finite.cpp
        backends/cuda/test_cuda_fractional.cpp
        backends/cuda/test_cuda_fusion.cpp
        backends/cuda/test_cuda_fusion_lifetime.cpp
        backends/cuda/test_cuda_linalg.cpp
        backends/cuda/test_cuda_memory.cpp
        backends/cuda/test_cuda_optimizer.cpp
        backends/cuda/test_cuda_plan.cpp
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
    )
    add_dependencies(
        ormi_cuda_backend_tests
        ormi_cuda_cache_helper
    )
    target_compile_definitions(
        ormi_cuda_backend_tests
        PRIVATE
            ORMI_CUDA_CACHE_HELPER="$<TARGET_FILE:ormi_cuda_cache_helper>"
    )
    ormi_stage_runtime(ormi_cuda_backend_tests)
    gtest_discover_tests(
        ormi_cuda_backend_tests
        PROPERTIES ENVIRONMENT
            "ORMI_CUDA_JIT_CACHE_DIR=${CMAKE_CURRENT_BINARY_DIR}/cuda-jit-cache-tests"
    )

    add_executable(ormi_cuda_state_tests
        helpers/main.cpp
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
    target_compile_definitions(
        ormi_cuda_state_tests
        PRIVATE ORMI_CUDA_BACKEND_BUILD
    )
    target_include_directories(ormi_cuda_state_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
    )
    target_link_libraries(ormi_cuda_state_tests PRIVATE
        ormi::core
        CUDA::cudart
        CUDA::cublas
        CUDA::cuda_driver
        GTest::gtest
    )
    ormi_stage_runtime(ormi_cuda_state_tests)
    gtest_discover_tests(ormi_cuda_state_tests)
endif()
