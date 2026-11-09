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
    public/op.cpp
    public/setup.cpp
)
target_compile_features(ormi_api_test PRIVATE cxx_std_20)
target_link_libraries(ormi_api_test PRIVATE ormi::core)
ormi_stage_runtime(ormi_api_test)
add_test(NAME ApiTest COMMAND ormi_api_test)

set(ORMI_RUNTIME_TEST_SRCS
    nn/modules/test_module.cpp
    nn/modules/test_state.cpp
    nn/modules/test_activation.cpp
    nn/modules/test_linear.cpp
    nn/modules/test_normalization.cpp
    nn/modules/test_batch_norm.cpp
    nn/modules/test_dropout.cpp
    nn/test_state.cpp
    nn/modules/test_batch_norm.cpp
    nn/modules/test_dropout.cpp
    nn/test_state.cpp
    nn/losses/test_mse.cpp
    nn/losses/test_classification.cpp
    nn/test_training.cpp
    core/test_autocast.cpp
    core/test_dtype.cpp
    core/test_float16.cpp
    core/test_shape.cpp
    core/test_setup.cpp
    core/test_tensor.cpp
    core/test_tensor_copy.cpp
    core/test_autograd.cpp
    core/test_training.cpp
    core/autograd/test_fractional.cpp
    core/autograd/test_linalg.cpp
    core/autograd/test_partial.cpp
    core/autograd/test_reduction.cpp
    core/autograd/test_view.cpp
    core/ops/test_cast.cpp
    core/ops/test_index.cpp
    core/ops/test_select.cpp
    core/ops/test_creation.cpp
    core/ops/test_finite.cpp
    core/ops/test_fractional.cpp
    core/ops/test_fractional_accuracy.cpp
    core/ops/test_linalg.cpp
    core/ops/test_pointwise.cpp
    core/ops/test_rsqrt.cpp
    core/ops/test_reduction.cpp
    core/ops/test_view.cpp
    core/optim/test_grad_scaler.cpp
    core/optim/test_sgd.cpp
    memory/test_allocator.cpp
    memory/test_buffer.cpp
    memory/test_ops.cpp
    platform/test_cpu.cpp
    platform/test_library.cpp
    platform/test_process.cpp
    runtime/test_backends.cpp
    runtime/test_device.cpp
    runtime/test_finite.cpp
    runtime/test_lower.cpp
    runtime/test_schedule.cpp
    runtime/test_step.cpp
    core/test_plan.cpp
    runtime/test_runtime_dispatch.cpp
    runtime/fusion/test_expr.cpp
    runtime/fusion/test_group.cpp
    runtime/fusion/test_key.cpp
    runtime/fusion/test_layout.cpp
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND ORMI_RUNTIME_TEST_SRCS platform/test_linux_process.cpp)
endif()

set(ORMI_BACKEND_TEST_SRCS
    backends/cpu/test_cpu_backend.cpp
    backends/cpu/test_pointwise.cpp
    backends/cpu/jit/test_accuracy.cpp
    backends/cpu/jit/test_cache.cpp
    backends/cpu/jit/test_codegen.cpp
    backends/cpu/jit/test_compile.cpp
    backends/cpu/jit/test_compiler.cpp
    backends/cpu/jit/test_compatibility.cpp
    backends/cpu/jit/test_execution.cpp
    backends/cpu/jit/test_input.cpp
    backends/cpu/jit/test_kernel.cpp
    backends/cpu/jit/test_loop.cpp
    backends/cpu/jit/test_source.cpp
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
add_test(
    NAME CPUJITAutomaticCompilerFallbackTest
    COMMAND ormi_jit_compiler_helper
        absent
        "${ORMI_TEST_JIT_COMPILER_CACHE}"
)
foreach(cache_state missing empty)
    add_test(
        NAME CPUJITLaterPublication_${cache_state}
        COMMAND ormi_jit_compiler_helper
            publish-${cache_state}
            "${ORMI_TEST_JIT_COMPILER_CACHE}"
            "${CMAKE_CXX_COMPILER}"
            "${ORMI_TEST_CXX_ID}"
    )
endforeach()

add_executable(ormi_process_helper
    helpers/process.cpp
)
target_compile_features(ormi_process_helper PRIVATE cxx_std_20)
if(MINGW)
    # Environment-replacement tests run this helper without PATH.
    target_link_options(ormi_process_helper PRIVATE -static)
endif()

add_executable(ormi_compiler_probe_helper helpers/compiler_probe.cpp)
target_compile_features(ormi_compiler_probe_helper PRIVATE cxx_std_20)
target_link_libraries(ormi_compiler_probe_helper PRIVATE Threads::Threads)

add_library(ormi_library_fixture SHARED helpers/library.cpp)
target_compile_features(ormi_library_fixture PRIVATE cxx_std_20)
ormi_configure_plugin(ormi_library_fixture)

add_dependencies(
    ormi_runtime_tests
    ormi_jit_cache_helper
    ormi_jit_compiler_helper
    ormi_process_helper
    ormi_compiler_probe_helper
    ormi_library_fixture
)
target_compile_definitions(ormi_runtime_tests PRIVATE
    ORMI_JIT_CACHE_HELPER="$<TARGET_FILE:ormi_jit_cache_helper>"
    ORMI_PROCESS_HELPER="$<TARGET_FILE:ormi_process_helper>"
    ORMI_COMPILER_PROBE_HELPER="$<TARGET_FILE:ormi_compiler_probe_helper>"
    ORMI_LIBRARY_FIXTURE="$<TARGET_FILE:ormi_library_fixture>"
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

add_library(ormi_jit_artifact_fixture SHARED
    helpers/jit_artifact.cpp
)
add_library(ormi_jit_artifact_incomplete SHARED
    helpers/jit_artifact.cpp
)
target_compile_definitions(ormi_jit_artifact_incomplete PRIVATE
    ORMI_TEST_INCOMPLETE_ARTIFACT
)

# Replace only cache acquisition in this target to force the lookup/acquisition race.
add_executable(ormi_jit_artifact_tests
    helpers/main.cpp
    backends/cpu/jit/test_artifact.cpp
    ../backends/cpu/jit/artifact.cpp
)
target_compile_features(ormi_jit_artifact_tests PRIVATE cxx_std_20)
target_include_directories(ormi_jit_artifact_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_link_libraries(ormi_jit_artifact_tests PRIVATE
    ormi::core
    GTest::gtest
)
add_dependencies(ormi_jit_artifact_tests
    ormi_jit_artifact_fixture
    ormi_jit_artifact_incomplete
)
target_compile_definitions(ormi_jit_artifact_tests PRIVATE
    ORMI_JIT_ARTIFACT_FIXTURE="$<TARGET_FILE:ormi_jit_artifact_fixture>"
    ORMI_JIT_ARTIFACT_INCOMPLETE="$<TARGET_FILE:ormi_jit_artifact_incomplete>"
)
ormi_stage_runtime(ormi_jit_artifact_tests)

if(ORMI_ENABLE_CUDA)
    target_compile_definitions(ormi_runtime_tests PRIVATE ORMI_ENABLE_CUDA)
endif()
ormi_stage_runtime(ormi_runtime_tests)

include(GoogleTest)

add_executable(ormi_executor_helper helpers/executor.cpp)
target_compile_features(ormi_executor_helper PRIVATE cxx_std_20)
target_include_directories(ormi_executor_helper PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(ormi_executor_helper PRIVATE ormi::core Threads::Threads)
ormi_stage_runtime(ormi_executor_helper)

add_executable(ormi_cpu_executor_tests
    backends/cpu/test_executor.cpp
    backends/cpu/executor/test_policy.cpp
    backends/cpu/executor/test_pool.cpp
)
target_compile_features(ormi_cpu_executor_tests PRIVATE cxx_std_20)
target_include_directories(ormi_cpu_executor_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(ormi_cpu_executor_tests PRIVATE
    ormi::core GTest::gtest_main Threads::Threads
)
target_compile_definitions(ormi_cpu_executor_tests PRIVATE
    ORMI_EXECUTOR_HELPER="$<TARGET_FILE:ormi_executor_helper>"
)
add_dependencies(ormi_cpu_executor_tests ormi_executor_helper)
ormi_stage_runtime(ormi_cpu_executor_tests)
gtest_discover_tests(ormi_cpu_executor_tests PROPERTIES
    TIMEOUT 30
    ENVIRONMENT "ORMI_CPU_THREADS=4"
)

# Replace process launch only here to check invocation arguments and failure handling.
add_executable(ormi_jit_invocation_tests
    backends/cpu/jit/test_invocation.cpp
    ../backends/cpu/jit/compiler/invoke.cpp
)
target_compile_features(ormi_jit_invocation_tests PRIVATE cxx_std_20)
target_include_directories(ormi_jit_invocation_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_link_libraries(ormi_jit_invocation_tests PRIVATE ormi::core GTest::gtest_main)
ormi_stage_runtime(ormi_jit_invocation_tests)
gtest_discover_tests(ormi_jit_invocation_tests PROPERTIES TIMEOUT 30)

# Replace process launch only here to verify probe cleanup and exact diagnostics.
add_executable(ormi_jit_probe_tests
    backends/cpu/jit/test_probe.cpp
    ../backends/cpu/jit/compiler/probe.cpp
)
target_compile_features(ormi_jit_probe_tests PRIVATE cxx_std_20)
target_include_directories(ormi_jit_probe_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_link_libraries(ormi_jit_probe_tests PRIVATE ormi::core GTest::gtest_main)
ormi_stage_runtime(ormi_jit_probe_tests)
gtest_discover_tests(ormi_jit_probe_tests PROPERTIES TIMEOUT 30)

add_test(NAME CPUJITArtifactWithoutCompilerTest
    COMMAND ormi_jit_artifact_tests
        --gtest_filter=CPUJITArtifactTest.loadsInitialCacheHitWithoutBuilding
)
set_tests_properties(CPUJITArtifactWithoutCompilerTest PROPERTIES
    TIMEOUT 30
    ENVIRONMENT
        "ORMI_CPU_JIT_CXX=${CMAKE_CURRENT_BINARY_DIR}/missing-compiler;ORMI_CPU_JIT_CXX_ID=gcc"
)
gtest_discover_tests(ormi_jit_artifact_tests
    PROPERTIES
        TIMEOUT 30
        ENVIRONMENT
            "ORMI_CPU_JIT_CXX=${CMAKE_CXX_COMPILER};ORMI_CPU_JIT_CXX_ID=${ORMI_TEST_CXX_ID};ORMI_CPU_JIT_CACHE_DIR=${CMAKE_CURRENT_BINARY_DIR}/jit-artifact-cache"
)
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
    # Replace driver calls only here to exercise every module-load failure step.
    add_executable(ormi_cuda_module_tests
        helpers/main.cpp
        ../backends/cuda/check.cu
        ../backends/cuda/fusion/module.cu
        backends/cuda/test_cuda_fusion_module.cpp
    )
    set_target_properties(ormi_cuda_module_tests PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )
    target_compile_features(ormi_cuda_module_tests PRIVATE cxx_std_20)
    target_include_directories(ormi_cuda_module_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
    )
    target_link_libraries(ormi_cuda_module_tests PRIVATE
        ormi::core
        CUDA::cudart
        GTest::gtest
    )
    ormi_stage_runtime(ormi_cuda_module_tests)
    gtest_discover_tests(ormi_cuda_module_tests
        PROPERTIES
            TIMEOUT 30
            ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1"
    )

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

    add_executable(ormi_cuda_nvrtc_tests
        helpers/main.cpp
        ../backends/cuda/fusion/nvrtc.cu
        backends/cuda/test_cuda_fusion_nvrtc.cpp
    )
    set_target_properties(ormi_cuda_nvrtc_tests PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )
    target_compile_features(ormi_cuda_nvrtc_tests PRIVATE cxx_std_20)
    target_include_directories(ormi_cuda_nvrtc_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
    )
    target_link_libraries(ormi_cuda_nvrtc_tests PRIVATE
        ormi::core
        CUDA::nvrtc
        GTest::gtest
    )
    ormi_stage_runtime(ormi_cuda_nvrtc_tests)
    gtest_discover_tests(ormi_cuda_nvrtc_tests
        PROPERTIES
            TIMEOUT 60
            ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1"
    )

    add_executable(ormi_cuda_backend_tests
        helpers/main.cpp
        ../backends/cuda/fusion/cache.cpp
        ../backends/cuda/fusion/codegen.cpp
        ../backends/cuda/fusion/codegen/reduction.cpp
        ../backends/cuda/fusion/codegen/value.cpp
        ../backends/cuda/fusion/nvrtc.cu
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
        backends/cuda/test_cuda_fusion_codegen_reduction.cpp
        backends/cuda/test_cuda_fusion_codegen_value.cpp
        backends/cuda/test_cuda_fusion_compile.cpp
        backends/cuda/test_cuda_fusion_execution.cpp
        backends/cuda/test_cuda_fusion_input.cpp
        backends/cuda/test_cuda_fusion_lifetime.cpp
        backends/cuda/test_cuda_linalg.cpp
        backends/cuda/test_cuda_memory.cpp
        backends/cuda/test_cuda_optimizer.cpp
        backends/cuda/test_cuda_plan.cpp
        backends/cuda/test_cuda_pointwise.cpp
        backends/cuda/test_cuda_nn.cpp
        backends/cuda/test_cuda_select.cpp
        backends/cuda/test_cuda_record.cpp
        backends/cuda/test_cuda_reduction.cpp
    )
    set_target_properties(ormi_cuda_backend_tests PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
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
        CUDA::nvrtc
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
