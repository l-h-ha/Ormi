if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)
FetchContent_Declare(
    benchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.8.5.zip
)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

if(TARGET benchmark)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(benchmark PRIVATE -Wno-switch)
    elseif(MSVC)
        target_compile_options(benchmark PRIVATE /wd4062)
    endif()
endif()

add_executable(ormi_runtime_bench
    main.cpp
    nn/modules/bench_linear.cpp
    nn/modules/bench_normalization.cpp
    nn/modules/bench_stateful.cpp
    nn/bench_functional.cpp
    graph/bench_autograd.cpp
    graph/bench_partial.cpp
    graph/bench_cast.cpp
    graph/bench_add.cpp
    graph/bench_unary.cpp
    graph/bench_matmul.cpp
    graph/bench_reduction.cpp
    graph/bench_view.cpp
    graph/bench_creation.cpp
    graph/bench_finite.cpp
    graph/bench_fractional.cpp
    backends/cpu/bench_executor.cpp
    backends/cpu/bench_matmul.cpp
    backends/cpu/bench_pointwise.cpp
    backends/cpu/bench_reduction.cpp
    memory/bench_memory.cpp
    optim/bench_grad_scaler.cpp
    optim/bench_sgd.cpp
    training/bench_dense.cpp
    training/bench_minibatch.cpp
    training/bench_step.cpp
    runtime/fusion/bench_build.cpp
    runtime/fusion/bench_expr.cpp
    runtime/fusion/bench_group.cpp
    runtime/fusion/bench_key.cpp
    runtime/fusion/bench_transcendental.cpp
    runtime/jit/bench_stages.cpp
    runtime/jit/bench_execution.cpp
    runtime/jit/bench_loop.cpp
    runtime/jit/bench_reduction.cpp
    runtime/bench_plan.cpp
    runtime/bench_runtime.cpp
    runtime/bench_schedule.cpp
    runtime/bench_step.cpp
)

target_compile_features(ormi_runtime_bench PRIVATE cxx_std_20)
target_include_directories(ormi_runtime_bench PRIVATE
    ${ORMI_GENERATED_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_link_libraries(ormi_runtime_bench PRIVATE
    ormi::core
    benchmark::benchmark
)
if(ORMI_CPU_HAS_AVX2_KERNELS)
    target_compile_definitions(
        ormi_runtime_bench
        PRIVATE ORMI_HAS_AVX2_KERNELS
    )
endif()
if(ORMI_ENABLE_CUDA)
    target_sources(ormi_runtime_bench PRIVATE
        backends/cuda/bench_allocator.cpp
        backends/cuda/bench_cast.cpp
        backends/cuda/bench_fill.cpp
        backends/cuda/bench_finite.cpp
        backends/cuda/bench_fractional.cpp
        backends/cuda/bench_fusion.cpp
        backends/cuda/bench_matmul.cpp
        backends/cuda/bench_memory.cpp
        backends/cuda/bench_optimizer.cpp
        backends/cuda/bench_pointwise.cpp
        backends/cuda/bench_reduction.cpp
    )
    target_compile_definitions(ormi_runtime_bench PRIVATE ORMI_ENABLE_CUDA)
    target_link_libraries(ormi_runtime_bench PRIVATE
        ormi::cuda
        CUDA::cudart
        CUDA::cuda_driver
    )
endif()
ormi_stage_runtime(ormi_runtime_bench)
