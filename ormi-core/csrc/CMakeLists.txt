add_compile_options($<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/FS>)

set(ORMI_CORE_SRCS
    core/autocast.cpp
    core/autograd/grad.cpp
    core/autograd/grads.cpp
    core/autograd/unbroadcast.cpp
    core/device.cpp
    core/dtype.cpp
    core/jit.cpp
    core/memory/buffer.cpp
    core/memory/ops.cpp
    core/optim/grad_scaler.cpp
    core/optim/sgd.cpp
    core/ops/cast.cpp
    core/ops/pointwise/binary.cpp
    core/ops/pointwise/unary.cpp
    core/ops/linalg.cpp
    core/ops/creation.cpp
    core/ops/reduction.cpp
    core/ops/view.cpp
    core/shape.cpp
    core/tensor.cpp
    core/runtime/backends.cpp
    core/runtime/fusion/expr.cpp
    core/runtime/realize.cpp
    core/runtime/dispatch/device.cpp
    core/runtime/dispatch/cast.cpp
    core/runtime/dispatch/finite.cpp
    core/runtime/dispatch/linalg.cpp
    core/runtime/dispatch/memory.cpp
    core/runtime/dispatch/optimizer.cpp
    core/runtime/dispatch/pointwise.cpp
    core/runtime/dispatch/reduction.cpp
    backends/cpu/allocator.cpp
    backends/cpu/backend.cpp
    backends/cpu/blas/openblas.cpp
    backends/cpu/cast.cpp
    backends/cpu/jit/cache.cpp
    backends/cpu/jit/codegen.cpp
    backends/cpu/jit/compiler.cpp
    backends/cpu/jit/kernel.cpp
    backends/cpu/matmul.cpp
    backends/cpu/memory.cpp
    backends/cpu/optimizer.cpp
    backends/cpu/pointwise/binary.cpp
    backends/cpu/pointwise/unary.cpp
    backends/cpu/reduction.cpp
    backends/cpu/reduction/finite.cpp
    backends/cpu/reduction/sum.cpp
)

if(WIN32)
    list(APPEND ORMI_CORE_SRCS platform/windows.cpp)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND ORMI_CORE_SRCS platform/linux.cpp)
else()
    message(FATAL_ERROR "ORMI currently supports Windows and Linux.")
endif()

option(ORMI_ENABLE_CUDA "Build the optional CUDA backend plugin." OFF)
option(ORMI_ENABLE_OPENBLAS "Use OpenBLAS for compatible CPU matmul operations." OFF)

if(ORMI_ENABLE_OPENBLAS)
    find_package(OpenBLAS 0.3.33 EXACT CONFIG REQUIRED)

    if(TARGET OpenBLAS::OpenBLAS)
        get_target_property(ORMI_OPENBLAS_IMPORTED OpenBLAS::OpenBLAS IMPORTED)
        if(ORMI_OPENBLAS_IMPORTED)
            set_property(TARGET OpenBLAS::OpenBLAS PROPERTY IMPORTED_GLOBAL TRUE)
        endif()
        set(ORMI_OPENBLAS_TARGET OpenBLAS::OpenBLAS)
    elseif(OpenBLAS_LIBRARY OR OpenBLAS_LIBRARIES)
        set(ORMI_OPENBLAS_LIBRARIES ${OpenBLAS_LIBRARY} ${OpenBLAS_LIBRARIES})
        list(REMOVE_DUPLICATES ORMI_OPENBLAS_LIBRARIES)
        add_library(ormi_openblas INTERFACE)
        target_link_libraries(ormi_openblas INTERFACE ${ORMI_OPENBLAS_LIBRARIES})
        if(OpenBLAS_INCLUDE_DIRS)
            target_include_directories(ormi_openblas INTERFACE ${OpenBLAS_INCLUDE_DIRS})
        elseif(OpenBLAS_INCLUDE_DIR)
            target_include_directories(ormi_openblas INTERFACE ${OpenBLAS_INCLUDE_DIR})
        endif()
        set(ORMI_OPENBLAS_TARGET ormi_openblas)
    else()
        message(FATAL_ERROR "The OpenBLAS package did not provide a library target or path.")
    endif()
endif()

add_library(ormi_core SHARED ${ORMI_CORE_SRCS})
add_library(ormi::core ALIAS ormi_core)
target_compile_features(ormi_core PUBLIC cxx_std_20)

set_target_properties(ormi_core PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
if(WIN32)
    set_target_properties(ormi_core PROPERTIES PREFIX "")
endif()

target_include_directories(ormi_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(ormi_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(ORMI_ENABLE_OPENBLAS)
    target_compile_definitions(ormi_core PRIVATE ORMI_ENABLE_OPENBLAS)
    target_link_libraries(ormi_core PRIVATE ${ORMI_OPENBLAS_TARGET})
    if(WIN32)
        if(NOT TARGET OpenBLAS::OpenBLAS)
            message(FATAL_ERROR "The Windows OpenBLAS package must provide OpenBLAS::OpenBLAS.")
        endif()
        add_custom_command(
            TARGET ormi_core
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:OpenBLAS::OpenBLAS>
                $<TARGET_FILE_DIR:ormi_core>
        )
    endif()
    message(STATUS "Enabled OpenBLAS.")
else()
    message(STATUS "OpenBLAS disabled.")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(ormi_core PUBLIC ${CMAKE_DL_LIBS})
endif()

if(ORMI_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_library(ormi_cuda_backend SHARED
        backends/cuda/allocator.cu
        backends/cuda/blas/cublas.cu
        backends/cuda/cast.cu
        backends/cuda/fill.cu
        backends/cuda/fusion/codegen.cpp
        backends/cuda/fusion/kernel.cu
        backends/cuda/matmul.cu
        backends/cuda/memory.cu
        backends/cuda/optimizer/sgd.cu
        backends/cuda/pointwise/binary.cu
        backends/cuda/pointwise/unary.cu
        backends/cuda/reduction.cpp
        backends/cuda/reduction/finite.cu
        backends/cuda/reduction/sum.cu
        backends/cuda/state.cu
        backends/cuda/backend.cpp
        backends/cuda/check.cu
    )
    add_library(ormi::cuda ALIAS ormi_cuda_backend)

    set_target_properties(ormi_cuda_backend PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )

    if(WIN32)
        set_target_properties(ormi_cuda_backend PROPERTIES PREFIX "")
    endif()

    target_compile_features(ormi_cuda_backend PRIVATE cxx_std_20)
    target_include_directories(ormi_cuda_backend PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(ormi_cuda_backend
        PRIVATE
            ormi::core
            CUDA::cudart
            CUDA::cublas
            CUDA::cuda_driver
            CUDA::nvrtc
    )
    message(STATUS "Enabled CUDA.")
else()
    message(STATUS "CUDA disabled.")
endif()

if(ORMI_BUILD_PYTHON)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)

    file(GLOB ORMI_BINDING_DEFS CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/bindings/defs/*.yaml
    )
    list(SORT ORMI_BINDING_DEFS)

    set(ORMI_BINDING_DIR ${CMAKE_CURRENT_BINARY_DIR}/bindings)
    set(ORMI_BINDING_CPP ${ORMI_BINDING_DIR}/generated.cpp)
    set(ORMI_BINDING_STUB_DIR ${ORMI_BINDING_DIR}/_C)
    set(ORMI_BINDING_STUBS
        ${ORMI_BINDING_STUB_DIR}/__init__.pyi
        ${ORMI_BINDING_STUB_DIR}/jit.pyi
        ${ORMI_BINDING_STUB_DIR}/jit.py
        ${ORMI_BINDING_STUB_DIR}/ops.pyi
        ${ORMI_BINDING_STUB_DIR}/ops.py
        ${ORMI_BINDING_STUB_DIR}/optim.pyi
        ${ORMI_BINDING_STUB_DIR}/optim.py
        ${ORMI_BINDING_DIR}/py.typed
    )

    add_custom_command(
        OUTPUT ${ORMI_BINDING_CPP} ${ORMI_BINDING_STUBS}
        COMMAND
            ${Python_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/bindings/generate.py
            --cpp ${ORMI_BINDING_CPP}
            --stub-dir ${ORMI_BINDING_STUB_DIR}
            --marker ${ORMI_BINDING_DIR}/py.typed
            --defs ${ORMI_BINDING_DEFS}
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/bindings/generate.py
            ${ORMI_BINDING_DEFS}
        COMMENT "Generating ORMI Python bindings and stubs"
        VERBATIM
    )

    pybind11_add_module(_C MODULE
        bindings/glue.cpp
        bindings/module.cpp
        ${ORMI_BINDING_CPP}
    )
    target_include_directories(_C PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(_C PRIVATE ormi::core)

    set_target_properties(_C PROPERTIES
        BUILD_RPATH "$<TARGET_FILE_DIR:ormi_core>"
        INSTALL_RPATH "$ORIGIN"
    )
    set_target_properties(ormi_core PROPERTIES INSTALL_RPATH "$ORIGIN")

    install(TARGETS _C ormi_core
        LIBRARY DESTINATION ormi_core/_ext
        RUNTIME DESTINATION ormi_core/_ext
    )
    if(ORMI_ENABLE_CUDA)
        install(TARGETS ormi_cuda_backend
            LIBRARY DESTINATION ormi_core/_ext
            RUNTIME DESTINATION ormi_core/_ext
        )

        if(WIN32)
            file(GLOB ORMI_NVRTC_BUILTINS
                "${CUDAToolkit_BIN_DIR}/nvrtc-builtins*.dll"
                "${CUDAToolkit_BIN_DIR}/x64/nvrtc-builtins*.dll"
            )
        else()
            file(GLOB ORMI_NVRTC_BUILTINS
                "${CUDAToolkit_LIBRARY_DIR}/libnvrtc-builtins.so*"
            )
        endif()
        if(NOT ORMI_NVRTC_BUILTINS)
            message(FATAL_ERROR "Cannot package CUDA: NVRTC builtins were not found.")
        endif()
        install(FILES ${ORMI_NVRTC_BUILTINS} DESTINATION ormi_core.libs)
    endif()
    if(WIN32 AND ORMI_ENABLE_OPENBLAS)
        install(FILES $<TARGET_FILE:OpenBLAS::OpenBLAS> DESTINATION ormi_core/_ext)
    endif()

    install(DIRECTORY ${ORMI_BINDING_STUB_DIR}/ DESTINATION ormi_core/_ext/_C)
    install(FILES ${ORMI_BINDING_DIR}/py.typed DESTINATION ormi_core)
endif()

function(ormi_stage_runtime target)
    if(WIN32)
        string(MD5 stage_id "${CMAKE_CURRENT_BINARY_DIR}")
        set(stage_target "ormi_runtime_stage_${stage_id}")

        if(NOT TARGET ${stage_target})
            set(stage_stamp "${CMAKE_CURRENT_BINARY_DIR}/ormi_runtime_stage.stamp")
            set(stage_dependencies ormi_core)
            set(stage_commands
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_FILE:ormi_core>
                    ${CMAKE_CURRENT_BINARY_DIR}
            )

            if(ORMI_ENABLE_CUDA)
                list(APPEND stage_dependencies ormi_cuda_backend)
                list(APPEND stage_commands
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        $<TARGET_FILE:ormi_cuda_backend>
                        ${CMAKE_CURRENT_BINARY_DIR}
                )
            endif()

            if(ORMI_ENABLE_OPENBLAS)
                list(APPEND stage_dependencies OpenBLAS::OpenBLAS)
                list(APPEND stage_commands
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        $<TARGET_FILE:OpenBLAS::OpenBLAS>
                        ${CMAKE_CURRENT_BINARY_DIR}
                )
            endif()

            add_custom_command(
                OUTPUT ${stage_stamp}
                ${stage_commands}
                COMMAND ${CMAKE_COMMAND} -E touch ${stage_stamp}
                DEPENDS ${stage_dependencies}
                VERBATIM
            )
            add_custom_target(${stage_target} DEPENDS ${stage_stamp})
        endif()

        add_dependencies(${target} ${stage_target})
    endif()
endfunction()

if(ORMI_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(ORMI_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
