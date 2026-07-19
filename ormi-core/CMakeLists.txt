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

if(ORMI_BUILD_TESTS)
    include(CTest)
    enable_testing()
endif()

add_subdirectory(csrc)

if(ORMI_BUILD_EXAMPLES)
    add_subdirectory(example)
endif()
