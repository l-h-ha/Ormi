if(NOT ORMI_SOURCE_DIR)
    message(FATAL_ERROR "ORMI_SOURCE_DIR is required.")
endif()

if(ORMI_TEST_UNSUPPORTED)
    set(CMAKE_SYSTEM_NAME UnknownOS)
    include(${ORMI_SOURCE_DIR}/cmake/platform.cmake)
    ormi_platform(
        NAME test
        SYSTEM TestOS
        SOURCES platform/test.cpp
    )
    ormi_require_platform()
    return()
endif()

set(CMAKE_SYSTEM_NAME TestOS)
include(${ORMI_SOURCE_DIR}/cmake/platform.cmake)

ormi_platform(
    NAME foreign
    SYSTEM OtherOS
    SOURCES platform/foreign.cpp
)
ormi_platform(
    NAME test
    SYSTEM TestOS
    SOURCES
        platform/test.cpp
        platform/test_process.cpp
    NO_PREFIX
    STAGE
    PLUGIN_RPATH test-origin
    LINKS test-loader
)

ormi_platform_name(name)
ormi_platform_sources(sources)
ormi_platform_stages(stages)
get_property(no_prefix GLOBAL PROPERTY ORMI_PLATFORM_NO_PREFIX)
get_property(rpath GLOBAL PROPERTY ORMI_PLATFORM_PLUGIN_RPATH)
get_property(links GLOBAL PROPERTY ORMI_PLATFORM_LINKS)

if(NOT name STREQUAL "test")
    message(FATAL_ERROR "The matching platform was not selected.")
endif()
if(NOT sources STREQUAL "platform/test.cpp;platform/test_process.cpp")
    message(FATAL_ERROR "The selected platform sources were not retained.")
endif()
if(NOT stages)
    message(FATAL_ERROR "The selected platform staging policy was not retained.")
endif()
if(NOT no_prefix OR NOT rpath STREQUAL "test-origin" OR NOT links STREQUAL "test-loader")
    message(FATAL_ERROR "The selected platform target traits were not retained.")
endif()
