add_executable(example
    benchmark.cpp
    example_1.cpp
    benchmark_2.cpp
)

target_link_libraries(example PRIVATE ormi_engine)