option(ORMI_ENABLE_CUDA "Build the optional CUDA backend plugin." OFF)

set(cuda_target_args)
if(ORMI_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    find_path(ORMI_CURAND_INCLUDE_DIR curand_kernel.h
        HINTS ${CUDAToolkit_INCLUDE_DIRS}
        REQUIRED
    )

    add_library(ormi_cuda_backend SHARED
        backends/cuda/allocator.cu
        backends/cuda/backend.cpp
        backends/cuda/blas/cublas.cu
        backends/cuda/cast.cu
        backends/cuda/indexing.cu
        backends/cuda/check.cu
        backends/cuda/fill.cu
        backends/cuda/fractional.cu
        backends/cuda/fractional/l2.cu
        backends/cuda/fusion/cache.cpp
        backends/cuda/fusion/codegen.cpp
        backends/cuda/fusion/codegen/reduction.cpp
        backends/cuda/fusion/codegen/value.cpp
        backends/cuda/fusion/compile.cu
        backends/cuda/fusion/execute.cu
        backends/cuda/fusion/kernel.cu
        backends/cuda/fusion/module.cu
        backends/cuda/fusion/nvrtc.cu
        backends/cuda/fusion/order.cpp
        backends/cuda/fusion/tile.cpp
        backends/cuda/matmul.cu
        backends/cuda/memory.cu
        backends/cuda/optimizer/sgd.cu
        backends/cuda/pointwise/binary.cu
        backends/cuda/pointwise/unary.cu
        backends/cuda/record.cu
        backends/cuda/random.cu
        backends/cuda/reduction.cpp
        backends/cuda/reduction/finite.cu
        backends/cuda/reduction/sum.cu
        backends/cuda/reduction/max.cu
        backends/cuda/state.cu
    )
    add_library(ormi::cuda ALIAS ormi_cuda_backend)

    set_target_properties(ormi_cuda_backend PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )

    target_compile_features(ormi_cuda_backend PRIVATE cxx_std_20)
    target_compile_definitions(
        ormi_cuda_backend
        PRIVATE ORMI_CUDA_BACKEND_BUILD
    )
    target_include_directories(ormi_cuda_backend PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${ORMI_CURAND_INCLUDE_DIR}
    )
    target_link_libraries(ormi_cuda_backend
        PRIVATE
            ormi::core
            CUDA::cudart
            CUDA::cublas
            CUDA::cuda_driver
            CUDA::nvrtc
    )

    if(ORMI_BUILD_PYTHON)
        ormi_platform_name(platform_name)
        if(platform_name STREQUAL "windows")
            file(GLOB ORMI_NVRTC_BUILTINS
                "${CUDAToolkit_BIN_DIR}/nvrtc-builtins*.dll"
                "${CUDAToolkit_BIN_DIR}/x64/nvrtc-builtins*.dll"
            )
        elseif(platform_name STREQUAL "linux")
            file(GLOB ORMI_NVRTC_BUILTINS
                "${CUDAToolkit_LIBRARY_DIR}/libnvrtc-builtins.so*"
            )
        else()
            message(FATAL_ERROR
                "CUDA packaging is not implemented for '${platform_name}'."
            )
        endif()
        if(NOT ORMI_NVRTC_BUILTINS)
            message(FATAL_ERROR "Cannot package CUDA: NVRTC builtins were not found.")
        endif()
        install(
            FILES ${ORMI_NVRTC_BUILTINS}
            DESTINATION ormi_cuda.libs
            COMPONENT cuda
        )
    endif()

    set(cuda_target_args TARGET ormi_cuda_backend COMPONENT cuda)
    message(STATUS "Enabled CUDA.")
else()
    message(STATUS "CUDA disabled.")
endif()

ormi_backend(
    ID 1
    NAME cuda
    LIBRARY ormi_cuda_backend
    INDEXED
    ${cuda_target_args}
)
