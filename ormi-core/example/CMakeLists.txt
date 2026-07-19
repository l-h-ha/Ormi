add_executable(ormi_example example.cpp)
target_link_libraries(ormi_example PRIVATE ormi::core)
ormi_stage_runtime(ormi_example)
