#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <thread>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <csignal>
#endif

int main(int argc, char** argv) {
    const char* configured = std::getenv("ORMI_PROBE_ROOT");
    if (!configured || argc < 2) {
        return 97;
    }
    const std::filesystem::path root(configured);
    const std::filesystem::path source(argv[argc - 1]);
    const bool gnu = argc == 4 &&
        std::string_view(argv[1]) == "-std=c++20" &&
        std::string_view(argv[2]) == "-fsyntax-only";
    const bool msvc = argc == 5 &&
        std::string_view(argv[1]) == "/nologo" &&
        std::string_view(argv[2]) == "/std:c++20" &&
        std::string_view(argv[3]) == "/Zs";
    if (!gnu && !msvc) {
        std::cerr << "incorrect probe arguments\n";
        return 98;
    }

    std::ifstream input(source);
    const std::string text{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()
    };
    if (text.find("ormi_compiler_probe") == std::string::npos) {
        std::cerr << "missing probe source\n";
        return 99;
    }
    {
        std::ofstream calls(root / "calls.txt", std::ios::app);
        calls << source.string() << '\n';
        if (!calls) {
            return 96;
        }
    }

    std::string mode;
    std::ifstream configuration(root / "mode.txt");
    configuration >> mode;
    std::cout << "probe fixture output\n" << std::flush;
    if (mode == "wait") {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!std::filesystem::exists(root / "release")) {
            if (std::chrono::steady_clock::now() >= deadline) {
                return 95;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return 0;
    }
    if (mode == "crash") {
#if defined(_WIN32)
        RaiseException(0xe0000001ul, EXCEPTION_NONCONTINUABLE, 0, nullptr);
#else
        std::raise(SIGKILL);
#endif
        return 94;
    }
    if (mode == "noisy") {
        std::cout << std::string(6000, 'x');
        return 23;
    }
    return mode == "ok" ? 0 : 23;
}
