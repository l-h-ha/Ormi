cmake_minimum_required(VERSION 3.18...3.27)

project(ormi_core LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

###
###
###


###
### CUDA
###

option(ORMI_USE_CUDA "Build with NVIDIA CUDA hardware acceleration" OFF)

find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND)
    enable_language(CUDA)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    set(ORMI_USE_CUDA ON)
    add_compile_definitions(ORMI_USE_CUDA)

    message(STATUS "NVIDIA CUDA backend enabled.")
else()
    message(STATUS "CUDA not found.")
endif()

###
### Python / pybind11
###

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
    message(FATAL_ERROR "Could not find pybind11. Please ensure it is installed.")
endif()

file(TO_CMAKE_PATH "${PYBIND11_CMAKE_DIR}" PYBIND11_CMAKE_DIR)
message(STATUS "Locking Pybind11 to: ${PYBIND11_CMAKE_DIR}")

find_package(pybind11 CONFIG REQUIRED PATHS "${PYBIND11_CMAKE_DIR}" NO_DEFAULT_PATH)

###
### BLAS
###

if(WIN32)
    find_package(OpenBLAS CONFIG REQUIRED)
else()
    find_library(OPENBLAS_LIB NAMES openblas REQUIRED)
endif()

find_path(OPENBLAS_INCLUDE_DIR NAMES cblas.h PATH_SUFFIXES openblas REQUIRED)

###
### Dependencies
###

include(FetchContent)

FetchContent_Declare(
    magic_enum
    GIT_REPOSITORY https://github.com/Neargye/magic_enum.git
    GIT_TAG        v0.9.5
)

FetchContent_MakeAvailable(magic_enum)

###
### CPU srcs
###

set(ORMI_SRCS
    csrc/core/tapes/compute_tape.cpp
    csrc/core/tapes/tape.cpp
    csrc/core/tapes/zero_grad_tape.cpp

    csrc/core/optim/sgd.cpp

    csrc/core/ops/blas/matmul.cpp

    csrc/core/ops/reduce/sum.cpp

    csrc/core/ops/pointwise/add.cpp
    csrc/core/ops/pointwise/cos.cpp
    csrc/core/ops/pointwise/div.cpp
    csrc/core/ops/pointwise/exp.cpp
    csrc/core/ops/pointwise/log.cpp
    csrc/core/ops/pointwise/mul.cpp
    csrc/core/ops/pointwise/neg.cpp
    csrc/core/ops/pointwise/pow.cpp
    csrc/core/ops/pointwise/sin.cpp
    csrc/core/ops/pointwise/sub.cpp

    csrc/core/ops/view/transpose.cpp
    csrc/core/ops/view/reshape.cpp
    csrc/core/ops/view/expand.cpp

    csrc/core/adjoint.cpp
    csrc/core/allocator.cpp
    csrc/core/autograd.cpp
    csrc/core/dtype.cpp
    csrc/core/buffer.cpp
    csrc/core/node.cpp
    csrc/core/shape.cpp

    csrc/backend/graph_breakers.cpp
    csrc/backend/runtime.cpp

    csrc/backend/cpu/allocator.cpp
    csrc/backend/cpu/compiler.cpp
    csrc/backend/cpu/jit.cpp
    csrc/backend/cpu/optimizer.cpp
    csrc/backend/cpu/utils.cpp

    csrc/backend/cpu/kernels/blas/matmul.cpp
    csrc/backend/cpu/kernels/reduce/sum.cpp

    csrc/backend/cpu/lowerings/pointwise.cpp
)

###
### CUDA srcs
###

if(ORMI_USE_CUDA)
    list(APPEND ORMI_SRCS
        csrc/backend/cuda/allocator.cu
        csrc/backend/cuda/compiler.cu
        csrc/backend/cuda/jit.cu
        csrc/backend/cuda/optimizer.cu
        csrc/backend/cuda/utils.cu
        csrc/backend/cuda/config.cu

        csrc/backend/cuda/kernels/blas/matmul.cu
        csrc/backend/cuda/kernels/reduce/sum.cu

        csrc/backend/cuda/lowerings/pointwise.cpp

        csrc/core/tapes/cuda_graph_tape.cu
    )
endif()

###
###
###

add_library(ormi_engine OBJECT ${ORMI_SRCS})

set_target_properties(ormi_engine PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

if(MSVC)
    target_compile_options(ormi_engine PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:/Z7>
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=/Z7>
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=/FS>
    )

    set_property(
        TARGET ormi_engine
        PROPERTY MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
    )
endif()

target_link_libraries(ormi_engine PUBLIC
    magic_enum::magic_enum
)

if(WIN32)
    target_link_libraries(ormi_engine PUBLIC
        OpenBLAS::OpenBLAS
    )
else()
    target_link_libraries(ormi_engine PUBLIC
        ${OPENBLAS_LIB}
    )
endif()

target_include_directories(ormi_engine PRIVATE
    ${OPENBLAS_INCLUDE_DIR}
)

target_include_directories(ormi_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
)

if(ORMI_USE_CUDA)
    target_link_libraries(ormi_engine PUBLIC
        CUDA::cudart_static
        CUDA::nvrtc
        CUDA::cuda_driver
        CUDA::cublas
    )
endif()

###
### Bindings
###

set(ORMI_PYEXT_TARGET _C)

set(ORMI_PYBIND_SOURCES
    csrc/bindings/main.cpp
    csrc/bindings/numpy_buffer.cpp
)

set(ORMI_BINDING_GENERATOR
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/bindings/generate_bindings.py"
)

file(GLOB ORMI_BINDING_DEFS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/bindings/defs/*.yaml"
)

set(ORMI_GENERATED_BINDINGS_CPP
    "${CMAKE_CURRENT_BINARY_DIR}/generated/bindings_generated.cpp"
)

set(ORMI_PYTHON_SRC_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

set(ORMI_EXT_STUB_DIR
    "${ORMI_PYTHON_SRC_DIR}/ormi_core/_ext"
)

set(ORMI_C_STUB
    "${ORMI_EXT_STUB_DIR}/_C/__init__.pyi"
)

set(ORMI_C_OPS_STUB
    "${ORMI_EXT_STUB_DIR}/_C/ops.pyi"
)

set(ORMI_PY_TYPED
    "${ORMI_PYTHON_SRC_DIR}/ormi_core/py.typed"
)

add_custom_command(
    OUTPUT
        "${ORMI_GENERATED_BINDINGS_CPP}"
        "${ORMI_C_STUB}"
        "${ORMI_C_OPS_STUB}"
        "${ORMI_PY_TYPED}"

    COMMAND "${Python_EXECUTABLE}"
            "${ORMI_BINDING_GENERATOR}"
            --out "${ORMI_GENERATED_BINDINGS_CPP}"
            --stub-out "${ORMI_C_STUB}"
            --ops-stub-out "${ORMI_C_OPS_STUB}"
            --py-typed "${ORMI_PY_TYPED}"
            --defs ${ORMI_BINDING_DEFS}

    DEPENDS
        "${ORMI_BINDING_GENERATOR}"
        ${ORMI_BINDING_DEFS}

    COMMENT "Generating ORMI pybind11 bindings and Python stubs"
    VERBATIM
)

set(ORMI_GENERATED_PYBIND_SOURCES
    "${ORMI_GENERATED_BINDINGS_CPP}"
)

pybind11_add_module(${ORMI_PYEXT_TARGET}
    ${ORMI_PYBIND_SOURCES}
    ${ORMI_GENERATED_PYBIND_SOURCES}
)

target_link_libraries(${ORMI_PYEXT_TARGET} PRIVATE
    ormi_engine
)

if(MSVC)
    target_compile_options(${ORMI_PYEXT_TARGET} PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:/Z7>
    )

    set_property(
        TARGET ${ORMI_PYEXT_TARGET}
        PROPERTY MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
    )

    target_link_options(${ORMI_PYEXT_TARGET} PRIVATE
        /NODEFAULTLIB:libcmt.lib
    )
endif()

set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/ormi_core/_ext")

set_target_properties(${ORMI_PYEXT_TARGET} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}"

    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIR}"

    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIR}"

    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${OUTPUT_DIR}"

    LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${OUTPUT_DIR}"
)

###
### CUDA DLL copy for Windows
###

if(ORMI_USE_CUDA AND WIN32)
    file(GLOB_RECURSE CUDA_DLLS
        "${CUDAToolkit_BIN_DIR}/nvrtc64*.dll"
        "${CUDAToolkit_BIN_DIR}/nvrtc-builtins*.dll"
        "${CUDAToolkit_BIN_DIR}/cublas64*.dll"
        "${CUDAToolkit_BIN_DIR}/cublasLt64*.dll"
    )

    if(CUDA_DLLS)
        add_custom_command(TARGET ${ORMI_PYEXT_TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CUDA_DLLS}
            "$<TARGET_FILE_DIR:${ORMI_PYEXT_TARGET}>"
        )
    endif()
endif()

###
### Tests / examples
###

if(NOT SKBUILD)
    include(CTest)
    enable_testing()

    add_subdirectory(csrc/tests)
    add_subdirectory(example)
endif()
