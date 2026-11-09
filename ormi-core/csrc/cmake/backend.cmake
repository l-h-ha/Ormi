include_guard(GLOBAL)

function(ormi_backend)
    set(options BUILTIN INDEXED HOST)
    set(one_value_args ID NAME LIBRARY INIT HEADER TARGET COMPONENT)
    cmake_parse_arguments(ARG "${options}" "${one_value_args}" "" ${ARGN})

    if(ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Unknown backend arguments: ${ARG_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT DEFINED ARG_ID OR NOT ARG_NAME)
        message(FATAL_ERROR "A backend requires ID and NAME.")
    endif()
    if(NOT ARG_ID MATCHES "^[0-9]+$")
        message(FATAL_ERROR "Backend '${ARG_NAME}' ID must be a non-negative integer.")
    endif()

    get_property(ids GLOBAL PROPERTY ORMI_BACKEND_IDS)
    list(FIND ids "${ARG_ID}" id_position)
    if(NOT id_position EQUAL -1)
        message(FATAL_ERROR "Backend ID ${ARG_ID} is already registered.")
    endif()

    get_property(names GLOBAL PROPERTY ORMI_BACKEND_NAMES)
    list(FIND names "${ARG_NAME}" name_position)
    if(NOT name_position EQUAL -1)
        message(FATAL_ERROR "Backend '${ARG_NAME}' is already registered.")
    endif()

    if(ARG_BUILTIN)
        if(NOT ARG_INIT OR NOT ARG_HEADER OR ARG_LIBRARY OR ARG_TARGET)
            message(FATAL_ERROR
                "Built-in backend '${ARG_NAME}' requires INIT and HEADER only."
            )
        endif()
    elseif(NOT ARG_LIBRARY OR ARG_INIT OR ARG_HEADER)
        message(FATAL_ERROR
            "Plugin backend '${ARG_NAME}' requires LIBRARY and no INIT or HEADER."
        )
    endif()

    if(ARG_TARGET)
        if(NOT TARGET ${ARG_TARGET})
            message(FATAL_ERROR
                "Backend '${ARG_NAME}' target '${ARG_TARGET}' does not exist."
            )
        endif()
        if(NOT ARG_COMPONENT)
            message(FATAL_ERROR
                "Backend '${ARG_NAME}' target requires an install COMPONENT."
            )
        endif()
        if(NOT COMMAND ormi_configure_plugin)
            message(FATAL_ERROR "Platform configuration must precede backend registration.")
        endif()
        ormi_configure_plugin(${ARG_TARGET})
        set_property(GLOBAL APPEND PROPERTY ORMI_RUNTIME_TARGETS "${ARG_TARGET}")
        set_property(
            GLOBAL PROPERTY ORMI_BACKEND_${ARG_ID}_COMPONENT "${ARG_COMPONENT}"
        )
    elseif(ARG_COMPONENT)
        message(FATAL_ERROR
            "Backend '${ARG_NAME}' has an install COMPONENT but no target."
        )
    endif()

    set_property(GLOBAL APPEND PROPERTY ORMI_BACKEND_IDS "${ARG_ID}")
    set_property(GLOBAL APPEND PROPERTY ORMI_BACKEND_NAMES "${ARG_NAME}")
    foreach(field NAME LIBRARY INIT HEADER BUILTIN INDEXED HOST TARGET)
        set_property(
            GLOBAL PROPERTY ORMI_BACKEND_${ARG_ID}_${field} "${ARG_${field}}"
        )
    endforeach()
endfunction()

function(ormi_write_catalog input output)
    get_property(ids GLOBAL PROPERTY ORMI_BACKEND_IDS)
    list(LENGTH ids count)
    if(count EQUAL 0)
        message(FATAL_ERROR "No ORMI backends were registered.")
    endif()

    set(headers "")
    set(rows "")
    math(EXPR last_id "${count} - 1")

    foreach(id RANGE 0 ${last_id})
        list(FIND ids "${id}" id_position)
        if(id_position EQUAL -1)
            message(FATAL_ERROR
                "Backend IDs must be contiguous from 0. ID ${id} is missing."
            )
        endif()

        foreach(field NAME LIBRARY INIT HEADER BUILTIN INDEXED HOST)
            get_property(
                backend_${field}
                GLOBAL PROPERTY ORMI_BACKEND_${id}_${field}
            )
        endforeach()

        if(backend_BUILTIN)
            string(APPEND headers "#include \"${backend_HEADER}\"\n")
            set(backend_library "")
            set(backend_init "${backend_INIT}")
        else()
            set(backend_library "${backend_LIBRARY}")
            set(backend_init "nullptr")
        endif()

        if(backend_INDEXED)
            set(backend_indexed true)
        else()
            set(backend_indexed false)
        endif()
        if(backend_HOST)
            set(backend_host true)
        else()
            set(backend_host false)
        endif()

        string(APPEND rows
            "        Entry{\"${backend_NAME}\", \"${backend_library}\", "
            "${backend_init}, ${backend_indexed}, ${backend_host}},\n"
        )
    endforeach()

    get_filename_component(output_dir "${output}" DIRECTORY)
    file(MAKE_DIRECTORY "${output_dir}")
    set(ORMI_BACKEND_HEADERS "${headers}")
    set(ORMI_BACKEND_ROWS "${rows}")
    configure_file("${input}" "${output}" @ONLY)
endfunction()

function(ormi_install_backends destination)
    get_property(ids GLOBAL PROPERTY ORMI_BACKEND_IDS)
    foreach(id IN LISTS ids)
        get_property(target GLOBAL PROPERTY ORMI_BACKEND_${id}_TARGET)
        if(target)
            get_property(component GLOBAL PROPERTY ORMI_BACKEND_${id}_COMPONENT)
            install(TARGETS ${target}
                LIBRARY DESTINATION ${destination} COMPONENT ${component}
                RUNTIME DESTINATION ${destination} COMPONENT ${component}
            )
        endif()
    endforeach()
endfunction()

function(ormi_runtime_target target)
    if(NOT TARGET ${target})
        message(FATAL_ERROR "Runtime target '${target}' does not exist.")
    endif()
    set_property(GLOBAL APPEND PROPERTY ORMI_RUNTIME_TARGETS "${target}")
endfunction()

function(ormi_runtime_targets output)
    get_property(targets GLOBAL PROPERTY ORMI_RUNTIME_TARGETS)
    set(${output} ${targets} PARENT_SCOPE)
endfunction()
