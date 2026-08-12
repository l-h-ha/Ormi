target_sources(ormi_core PRIVATE
    backends/cpu/allocator.cpp
    backends/cpu/backend.cpp
    backends/cpu/blas/openblas.cpp
    backends/cpu/cast.cpp
    backends/cpu/executor.cpp
    backends/cpu/fractional.cpp
    backends/cpu/fractional/l1.cpp
    backends/cpu/fractional/l2.cpp
    backends/cpu/jit/artifact.cpp
    backends/cpu/jit/cache.cpp
    backends/cpu/jit/codegen.cpp
    backends/cpu/jit/codegen/map.cpp
    backends/cpu/jit/codegen/reduction.cpp
    backends/cpu/jit/codegen/value.cpp
    backends/cpu/jit/compile.cpp
    backends/cpu/jit/compiler.cpp
    backends/cpu/jit/execute.cpp
    backends/cpu/jit/kernel.cpp
    backends/cpu/jit/loop.cpp
    backends/cpu/jit/math.cpp
    backends/cpu/jit/target.cpp
    backends/cpu/matmul.cpp
    backends/cpu/memory.cpp
    backends/cpu/optimizer.cpp
    backends/cpu/pointwise/binary.cpp
    backends/cpu/pointwise/unary.cpp
    backends/cpu/reduction.cpp
    backends/cpu/reduction/finite.cpp
    backends/cpu/reduction/sum.cpp
)

set(ORMI_CPU_HAS_AVX2_KERNELS OFF)
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" ORMI_CPU_PROCESSOR)
if(ORMI_CPU_PROCESSOR MATCHES "^(amd64|x86_64|x64)$")
    if(MSVC OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
        set(ORMI_CPU_AVX2_FLAG /arch:AVX2)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
        set(ORMI_CPU_AVX2_FLAG -mavx2)
        set(ORMI_CPU_FMA_FLAG -mfma)
    endif()

    if(ORMI_CPU_AVX2_FLAG)
        set(ORMI_CPU_HAS_AVX2_KERNELS ON)
        target_sources(ormi_core PRIVATE
            backends/cpu/matmul/avx2.cpp
            backends/cpu/matmul/parallel.cpp
            backends/cpu/matmul/row_gemm.cpp
            backends/cpu/reduction/avx2.cpp
        )
        set_property(
            SOURCE
                backends/cpu/matmul.cpp
                backends/cpu/reduction/sum.cpp
            APPEND PROPERTY COMPILE_DEFINITIONS ORMI_HAS_AVX2_KERNELS
        )
        set_property(
            SOURCE
                backends/cpu/matmul/avx2.cpp
                backends/cpu/matmul/row_gemm.cpp
            APPEND PROPERTY COMPILE_OPTIONS
                ${ORMI_CPU_AVX2_FLAG}
                ${ORMI_CPU_FMA_FLAG}
        )
        set_property(
            SOURCE backends/cpu/reduction/avx2.cpp
            APPEND PROPERTY COMPILE_OPTIONS ${ORMI_CPU_AVX2_FLAG}
        )
    endif()
endif()
unset(ORMI_CPU_AVX2_FLAG)
unset(ORMI_CPU_FMA_FLAG)
unset(ORMI_CPU_PROCESSOR)

ormi_backend(
    ID 0
    NAME cpu
    BUILTIN
    HOST
    INIT ormi::backends::cpu::init
    HEADER backends/cpu/backend.h
)

option(ORMI_ENABLE_SLEEF "Use SLEEF for AVX2 CPU JIT transcendental operations." OFF)

if(ORMI_ENABLE_SLEEF)
    find_package(sleef 3.9.0 EXACT CONFIG REQUIRED)

    target_compile_definitions(ormi_core PRIVATE ORMI_ENABLE_SLEEF)
    target_link_libraries(ormi_core PRIVATE sleef::sleef)

    if(MSVC)
        set_property(
            SOURCE backends/cpu/jit/math.cpp
            APPEND PROPERTY COMPILE_OPTIONS /arch:AVX2
        )
    else()
        set_property(
            SOURCE backends/cpu/jit/math.cpp
            APPEND PROPERTY COMPILE_OPTIONS -mavx2
        )
    endif()

    ormi_runtime_target(sleef::sleef)
    ormi_platform_name(platform_name)
    if(ORMI_BUILD_PYTHON AND platform_name STREQUAL "windows")
        install(
            FILES $<TARGET_FILE:sleef::sleef>
            DESTINATION ormi_core/_ext
            COMPONENT core
        )
    endif()
    message(STATUS "Enabled SLEEF CPU vector math.")
else()
    message(STATUS "SLEEF CPU vector math disabled.")
endif()

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

    target_compile_definitions(ormi_core PRIVATE ORMI_ENABLE_OPENBLAS)
    target_link_libraries(ormi_core PRIVATE ${ORMI_OPENBLAS_TARGET})

    ormi_platform_name(platform_name)
    if(platform_name STREQUAL "windows")
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
        ormi_runtime_target(OpenBLAS::OpenBLAS)

        if(ORMI_BUILD_PYTHON)
            install(
                FILES $<TARGET_FILE:OpenBLAS::OpenBLAS>
                DESTINATION ormi_core/_ext
                COMPONENT core
            )
        endif()
    endif()
    message(STATUS "Enabled OpenBLAS.")
else()
    message(STATUS "OpenBLAS disabled.")
endif()
