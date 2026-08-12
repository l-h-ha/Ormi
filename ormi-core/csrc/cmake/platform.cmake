include_guard(GLOBAL)

function(ormi_platform)
    set(options EXPORT_ALL NO_PREFIX STAGE)
    set(one_value_args NAME SYSTEM PLUGIN_RPATH)
    set(multi_value_args SOURCES LINKS)
    cmake_parse_arguments(
        ARG
        "${options}"
        "${one_value_args}"
        "${multi_value_args}"
        ${ARGN}
    )

    if(ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Unknown platform arguments: ${ARG_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT ARG_NAME OR NOT ARG_SYSTEM OR NOT ARG_SOURCES)
        message(FATAL_ERROR "A platform requires NAME, SYSTEM, and SOURCES.")
    endif()

    get_property(names GLOBAL PROPERTY ORMI_PLATFORM_NAMES)
    list(FIND names "${ARG_NAME}" name_position)
    if(NOT name_position EQUAL -1)
        message(FATAL_ERROR "Platform '${ARG_NAME}' is already registered.")
    endif()
    set_property(GLOBAL APPEND PROPERTY ORMI_PLATFORM_NAMES "${ARG_NAME}")

    if(NOT "${CMAKE_SYSTEM_NAME}" STREQUAL "${ARG_SYSTEM}")
        return()
    endif()

    get_property(selected GLOBAL PROPERTY ORMI_PLATFORM)
    if(selected)
        message(FATAL_ERROR
            "Platforms '${selected}' and '${ARG_NAME}' both match ${CMAKE_SYSTEM_NAME}."
        )
    endif()

    set_property(GLOBAL PROPERTY ORMI_PLATFORM "${ARG_NAME}")
    foreach(field SOURCES EXPORT_ALL NO_PREFIX STAGE PLUGIN_RPATH LINKS)
        set_property(GLOBAL PROPERTY ORMI_PLATFORM_${field} "${ARG_${field}}")
    endforeach()
endfunction()

function(ormi_require_platform)
    get_property(selected GLOBAL PROPERTY ORMI_PLATFORM)
    if(selected)
        return()
    endif()

    get_property(names GLOBAL PROPERTY ORMI_PLATFORM_NAMES)
    list(JOIN names ", " supported)
    message(FATAL_ERROR
        "ORMI does not support '${CMAKE_SYSTEM_NAME}'. Registered platforms: ${supported}."
    )
endfunction()

function(ormi_configure_platform target)
    ormi_require_platform()
    get_property(sources GLOBAL PROPERTY ORMI_PLATFORM_SOURCES)
    get_property(export_all GLOBAL PROPERTY ORMI_PLATFORM_EXPORT_ALL)
    get_property(no_prefix GLOBAL PROPERTY ORMI_PLATFORM_NO_PREFIX)
    get_property(links GLOBAL PROPERTY ORMI_PLATFORM_LINKS)

    target_sources(${target} PRIVATE ${sources})
    if(export_all)
        set_target_properties(${target} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
    if(no_prefix)
        set_target_properties(${target} PROPERTIES PREFIX "")
    endif()
    if(links)
        target_link_libraries(${target} PUBLIC ${links})
    endif()
endfunction()

function(ormi_configure_plugin target)
    ormi_require_platform()
    get_property(no_prefix GLOBAL PROPERTY ORMI_PLATFORM_NO_PREFIX)
    get_property(rpath GLOBAL PROPERTY ORMI_PLATFORM_PLUGIN_RPATH)

    if(no_prefix)
        set_target_properties(${target} PROPERTIES PREFIX "")
    endif()
    if(rpath)
        set_target_properties(${target} PROPERTIES INSTALL_RPATH "${rpath}")
    endif()
endfunction()

function(ormi_platform_stages output)
    ormi_require_platform()
    get_property(stage GLOBAL PROPERTY ORMI_PLATFORM_STAGE)
    set(${output} ${stage} PARENT_SCOPE)
endfunction()

function(ormi_platform_name output)
    ormi_require_platform()
    get_property(selected GLOBAL PROPERTY ORMI_PLATFORM)
    set(${output} ${selected} PARENT_SCOPE)
endfunction()

function(ormi_platform_sources output)
    ormi_require_platform()
    get_property(sources GLOBAL PROPERTY ORMI_PLATFORM_SOURCES)
    set(${output} ${sources} PARENT_SCOPE)
endfunction()
