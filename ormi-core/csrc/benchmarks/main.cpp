#include <exception>
#include <iostream>

#include <benchmark/benchmark.h>

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);

    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }

    try {
        benchmark::RunSpecifiedBenchmarks();
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: " << error.what() << '\n';
        benchmark::Shutdown();
        return 1;
    }

    benchmark::Shutdown();
}
