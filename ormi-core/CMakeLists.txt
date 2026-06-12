cmake_minimum_required(VERSION 3.18...3.27)
project(ormi_core LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(Python_FIND_REGISTRY NEVER)
set(Python_FIND_FRAMEWORK NEVER)
set(Python_FIND_STRATEGY LOCATION)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT PYBIND11_CMAKE_DIR)
    message(FATAL_ERROR "Could not find pybind11. Ensure it is installed in the active virtual environment.")
endif()

file(TO_CMAKE_PATH "${PYBIND11_CMAKE_DIR}" PYBIND11_CMAKE_DIR)

message(STATUS "Strictly locking Pybind11 to: ${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED PATHS "${PYBIND11_CMAKE_DIR}" NO_DEFAULT_PATH)

find_package(OpenBLAS CONFIG REQUIRED)
find_path(OPENBLAS_INCLUDE_DIR NAMES cblas.h PATH_SUFFIXES openblas)

find_package(CUDAToolkit REQUIRED)

add_library(ormi_engine OBJECT 
    csrc/core/memory.cpp
    csrc/backend/cpu/compiler.cpp
    csrc/backend/cuda/compiler.cu
    csrc/core/autograd.cpp
    csrc/core/adjoint.cpp
    csrc/core/shape.cpp
    csrc/backend/cuda/kernels/blas/matmul.cu
    csrc/backend/cuda/utils.cu
    csrc/core/allocator.cpp
    csrc/backend/cpu/allocator.cpp
    csrc/backend/cuda/allocator.cu
    csrc/backend/runtime.h
    csrc/backend/runtime.cpp
    csrc/backend/cpu/jit.h
    csrc/backend/cpu/jit.cpp
    csrc/backend/cuda/jit.cu
    csrc/backend/graph_breakers.h
    csrc/backend/graph_breakers.cpp
    csrc/backend/cpu/kernels/blas/cpu_matmul.h
    csrc/backend/cpu/kernels/blas/matmul.cpp
    csrc/core/node.cpp
    csrc/core/attributes.h
    csrc/backend/cpu/kernels/reduce/sum.cpp
    csrc/backend/cpu/kernels/reduce/cpu_sum.h
    csrc/core/ops/blas/matmul.cpp
    csrc/core/ops/reduce/sum.cpp
    csrc/core/ops/blas/matmul.h
    csrc/core/ops/reduce/sum.h
    csrc/backend/cuda/kernels/reduce/sum.cu
    csrc/core/registry.h
    csrc/core/ops/pointwise/add.cpp
    csrc/core/ops/pointwise/mul.cpp
    csrc/core/ops/pointwise/sub.cpp
    csrc/core/ops/pointwise/div.cpp
    csrc/core/ops/pointwise/neg.cpp
    csrc/core/ops/pointwise/exp.cpp
    csrc/core/ops/pointwise/sin.cpp
    csrc/core/ops/pointwise/cos.cpp
    csrc/core/ops/pointwise/add.h
    csrc/core/ops/pointwise/cos.h
    csrc/core/ops/pointwise/div.h
    csrc/core/ops/pointwise/exp.h
    csrc/core/ops/pointwise/mul.h
    csrc/core/ops/pointwise/neg.h
    csrc/core/ops/pointwise/sin.h
    csrc/core/ops/pointwise/sub.h
    csrc/core/ops/view/transpose.cpp
    csrc/core/ops/view/transpose.h
    csrc/core/ops/view/reshape.cpp
    csrc/core/ops/view/reshape.h
    csrc/core/ops/view/expand.cpp
    csrc/core/ops/view/expand.h
    csrc/core/ops/pointwise/pow.cpp
    csrc/core/ops/pointwise/pow.h
    csrc/core/ops/pointwise/log.cpp
    csrc/core/ops/pointwise/log.h
    csrc/backend/cuda/lowering_registry.h
    csrc/backend/cuda/lowerings/pointwise.cpp
    csrc/backend/cpu/lowering_registry.h
    csrc/backend/cpu/lowerings/pointwise.cpp
)

target_link_libraries(ormi_engine PRIVATE 
    CUDA::cudart_static
    CUDA::nvrtc 
    CUDA::cuda_driver
    CUDA::cublas
)
target_link_libraries(ormi_engine PRIVATE OpenBLAS::OpenBLAS)
target_include_directories(ormi_engine PRIVATE ${OPENBLAS_INCLUDE_DIR})

target_include_directories(ormi_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/csrc)

pybind11_add_module(ormi_backend 
    csrc/bindings/main.cpp
    csrc/bindings/bind_blas.cpp
    csrc/bindings/bind_device.cpp
    csrc/bindings/bind_graph.cpp
    csrc/bindings/bind_memory.cpp
    csrc/bindings/bind_operation.cpp
    csrc/bindings/bind_pointwise.cpp
    csrc/bindings/bind_reduce.cpp
    csrc/bindings/bind_view.cpp
)
target_link_libraries(ormi_backend PRIVATE ormi_engine)

set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/ormi_core/_ext")
set_target_properties(ormi_backend PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIR}"
)

if(WIN32)
    file(GLOB_RECURSE CUDA_DLLS 
        "${CUDAToolkit_BIN_DIR}/nvrtc64*.dll"
        "${CUDAToolkit_BIN_DIR}/nvrtc-builtins*.dll"
        "${CUDAToolkit_BIN_DIR}/cublas64*.dll"
        "${CUDAToolkit_BIN_DIR}/cublasLt64*.dll"
    )

    if(CUDA_DLLS)
        add_custom_command(TARGET ormi_backend POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CUDA_DLLS}
            "$<TARGET_FILE_DIR:ormi_backend>"
        )
    endif()
endif()

if(MSVC)
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "Embedded")

    set_property(TARGET ormi_engine PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    set_property(TARGET ormi_backend PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    target_link_options(ormi_backend PRIVATE /NODEFAULTLIB:libcmt.lib)
endif()

enable_testing()
add_subdirectory(csrc/tests)