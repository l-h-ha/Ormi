cmake_minimum_required(VERSION 3.18)

if(POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
    set(
        CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
        "$<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:Embedded>"
    )
endif()

project(ormi LANGUAGES CXX)

option(ORMI_BUILD_TESTS "Build tests for the C++ runtime." ON)
option(ORMI_BUILD_BENCHMARKS "Build benchmarks for the C++ runtime." ON)
option(ORMI_BUILD_PYTHON "Build the generated Python extension." OFF)
option(ORMI_BUILD_EXAMPLES "Build C++ examples." OFF)
option(ORMI_ENABLE_SANITIZERS "Instrument CPU builds with compiler sanitizers." OFF)

if(ORMI_ENABLE_SANITIZERS)
    if(WIN32)
        if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            message(FATAL_ERROR "Windows sanitizer builds require MSVC.")
        endif()

        if(
            NOT CMAKE_SIZEOF_VOID_P EQUAL 8 OR
            NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64)$"
        )
            message(FATAL_ERROR "Windows sanitizer builds currently require x86-64.")
        endif()

        get_filename_component(ORMI_COMPILER_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
        get_filename_component(ORMI_COMPILER_HOST_DIR "${ORMI_COMPILER_DIR}" DIRECTORY)
        get_filename_component(ORMI_COMPILER_BIN_DIR "${ORMI_COMPILER_HOST_DIR}" DIRECTORY)
        get_filename_component(ORMI_VC_TOOLS_DIR "${ORMI_COMPILER_BIN_DIR}" DIRECTORY)
        file(TO_CMAKE_PATH "${ORMI_VC_TOOLS_DIR}" ORMI_VC_TOOLS_DIR)
        set(
            ORMI_SANITIZER_RUNTIME
            "${ORMI_VC_TOOLS_DIR}/bin/Hostx64/x64/clang_rt.asan_dynamic-x86_64.dll"
        )
        if(NOT EXISTS "${ORMI_SANITIZER_RUNTIME}")
            message(
                FATAL_ERROR
                "Cannot locate Visual Studio's x86-64 AddressSanitizer runtime. "
                "Install the C++ AddressSanitizer component and configure from "
                "a Visual Studio developer environment."
            )
        endif()

        add_compile_options(/fsanitize=address /Oy-)
        add_link_options(/INCREMENTAL:NO)
    else()
        if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            message(FATAL_ERROR "Linux sanitizer builds require Clang.")
        endif()

        add_compile_options(
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
            -fno-sanitize-recover=all
        )
        add_link_options(-fsanitize=address,undefined)
    endif()
endif()

if(ORMI_BUILD_TESTS)
    include(CTest)
    enable_testing()
endif()

add_subdirectory(csrc)

if(ORMI_BUILD_EXAMPLES)
    add_subdirectory(example)
endif()
