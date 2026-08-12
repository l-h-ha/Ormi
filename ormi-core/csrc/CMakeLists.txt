add_compile_options($<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/FS>)

set(ORMI_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)

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
    core/ops/fractional.cpp
    core/ops/fractional/nodes.cpp
    core/ops/fractional/weights.cpp
    core/ops/pointwise/binary.cpp
    core/ops/pointwise/unary.cpp
    core/ops/linalg.cpp
    core/ops/creation.cpp
    core/ops/reduction.cpp
    core/ops/view.cpp
    core/shape.cpp
    core/tensor.cpp
    core/runtime/backend.cpp
    core/runtime/backends.cpp
    core/runtime/fusion/expr.cpp
    core/runtime/fusion/group.cpp
    core/runtime/fusion/key.cpp
    core/runtime/lower.cpp
    core/runtime/plan.cpp
    core/runtime/realize.cpp
    core/runtime/schedule.cpp
    core/runtime/step.cpp
    core/runtime/dispatch/device.cpp
    core/runtime/dispatch/cast.cpp
    core/runtime/dispatch/finite.cpp
    core/runtime/dispatch/fractional.cpp
    core/runtime/dispatch/linalg.cpp
    core/runtime/dispatch/memory.cpp
    core/runtime/dispatch/optimizer.cpp
    core/runtime/dispatch/pointwise.cpp
    core/runtime/dispatch/reduction.cpp
)

include(cmake/platform.cmake)
file(GLOB ORMI_PLATFORM_FILES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/platform/*.cmake
)
list(SORT ORMI_PLATFORM_FILES)
foreach(platform_file IN LISTS ORMI_PLATFORM_FILES)
    include(${platform_file})
endforeach()
ormi_require_platform()

find_package(Threads REQUIRED)

add_library(ormi_core SHARED ${ORMI_CORE_SRCS})
add_library(ormi::core ALIAS ormi_core)
target_compile_features(ormi_core PUBLIC cxx_std_20)
target_link_libraries(ormi_core PRIVATE Threads::Threads)

ormi_configure_platform(ormi_core)

target_include_directories(ormi_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(ormi_core PRIVATE
    ${ORMI_GENERATED_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

include(cmake/backend.cmake)
file(GLOB ORMI_BACKEND_FILES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/*/backend.cmake
)
list(SORT ORMI_BACKEND_FILES)
foreach(backend_file IN LISTS ORMI_BACKEND_FILES)
    include(${backend_file})
endforeach()

ormi_write_catalog(
    ${CMAKE_CURRENT_SOURCE_DIR}/core/runtime/catalog.h.in
    ${ORMI_GENERATED_DIR}/core/runtime/catalog.h
)

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
        LIBRARY DESTINATION ormi_core/_ext COMPONENT core
        RUNTIME DESTINATION ormi_core/_ext COMPONENT core
    )
    ormi_install_backends(ormi_core/_ext)

    install(
        DIRECTORY ${ORMI_BINDING_STUB_DIR}/
        DESTINATION ormi_core/_ext/_C
        COMPONENT core
    )
    install(
        FILES ${ORMI_BINDING_DIR}/py.typed
        DESTINATION ormi_core
        COMPONENT core
    )
endif()

function(ormi_stage_runtime target)
    ormi_platform_stages(stage_runtime)
    if(stage_runtime)
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

            ormi_runtime_targets(runtime_targets)
            foreach(runtime_target IN LISTS runtime_targets)
                list(APPEND stage_dependencies ${runtime_target})
                list(APPEND stage_commands
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        $<TARGET_FILE:${runtime_target}>
                        ${CMAKE_CURRENT_BINARY_DIR}
                )
            endforeach()

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

    if(ORMI_SANITIZER_RUNTIME)
        add_custom_command(
            TARGET ${target}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ORMI_SANITIZER_RUNTIME}"
                "$<TARGET_FILE_DIR:${target}>"
            VERBATIM
        )
    endif()
endfunction()

if(ORMI_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(ORMI_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
