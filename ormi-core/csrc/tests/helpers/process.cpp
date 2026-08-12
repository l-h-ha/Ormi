#include <chrono>
#include <iostream>
#include <string_view>
#include <thread>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <csignal>
#endif

int main(int argc, char** argv) {
    if (argc != 2) {
        return 2;
    }

    const std::string_view mode(argv[1]);
    if (mode == "normal") {
        std::cout << "normal output\n";
        return 7;
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
