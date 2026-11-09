#include <chrono>
#include <cstdlib>
#include <iostream>
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
    if (argc < 2) {
        return 2;
    }

    const std::string_view mode(argv[1]);
    if (mode == "arguments") {
        for (int i = 2; i < argc; ++i) {
            const std::string_view argument(argv[i]);
            std::cout << argument.size() << ':' << argument << '\n';
        }
        return 0;
    }
    if (mode == "environment") {
        if (argc != 3) {
            return 2;
        }
        const auto* value = std::getenv(argv[2]);
        std::cout << (value ? value : "<unset>");
        return 0;
    }
    if (argc != 2) {
        return 2;
    }
    if (mode == "normal") {
        std::cout << "normal output\n";
        return 7;
    }
    if (mode == "quiet") {
        return 0;
    }
    if (mode == "streams") {
        std::cout << "stdout\n" << std::flush;
        std::cerr << "stderr\n";
        return 0;
    }
    if (mode == "large") {
        std::cout << std::string(128 * 1024, 'x') << std::flush;
        std::cerr << std::string(128 * 1024, 'y');
        return 17;
    }
    if (mode == "sleep") {
        std::cout << "before timeout\n" << std::flush;
        std::this_thread::sleep_for(std::chrono::seconds(10));
        return 0;
    }
    if (mode == "crash") {
        std::cout << "before crash\n" << std::flush;
#if defined(_WIN32)
        RaiseException(0xe0000001ul, EXCEPTION_NONCONTINUABLE, 0, nullptr);
#else
        std::raise(SIGKILL);
#endif
        return 3;
    }
    return 2;
}
