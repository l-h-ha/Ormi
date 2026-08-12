if(NOT ORMI_SOURCE_DIR OR NOT ORMI_TEST_DIR)
    message(FATAL_ERROR "ORMI_SOURCE_DIR and ORMI_TEST_DIR are required.")
endif()

include(${ORMI_SOURCE_DIR}/cmake/backend.cmake)

ormi_backend(
    ID 1
    NAME accelerator
    LIBRARY ormi_accelerator_backend
    INDEXED
)
ormi_backend(
    ID 0
    NAME host
    BUILTIN
    HOST
    INIT ormi::backends::host::init
    HEADER backends/host/backend.h
)

set(output ${ORMI_TEST_DIR}/core/runtime/catalog.h)
ormi_write_catalog(
    ${ORMI_SOURCE_DIR}/core/runtime/catalog.h.in
    ${output}
)
file(READ ${output} catalog)

set(host_row
    "Entry{\"host\", \"\", ormi::backends::host::init, false, true}"
)
set(accelerator_row
    "Entry{\"accelerator\", \"ormi_accelerator_backend\", nullptr, true, false}"
)

string(FIND "${catalog}" "${host_row}" host_position)
string(FIND "${catalog}" "${accelerator_row}" accelerator_position)
if(host_position EQUAL -1 OR accelerator_position EQUAL -1)
    message(FATAL_ERROR "Generated backend rows do not match their declarations.")
endif()
if(NOT host_position LESS accelerator_position)
    message(FATAL_ERROR "Generated backend rows are not ordered by stable ID.")
endif()
