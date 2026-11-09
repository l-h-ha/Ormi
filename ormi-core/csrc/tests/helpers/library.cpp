#if defined(_WIN32)
#define ORMI_TEST_EXPORT __declspec(dllexport)
#else
#define ORMI_TEST_EXPORT __attribute__((visibility("default")))
#endif

namespace {
    unsigned* unloaded = nullptr;

    struct Unload {
        ~Unload() {
            if (unloaded) {
                ++*unloaded;
            }
        }
    } unload;
}

extern "C" ORMI_TEST_EXPORT int ormi_library_value() {
    return 42;
}

extern "C" ORMI_TEST_EXPORT void ormi_library_watch_unload(unsigned* counter) {
    unloaded = counter;
}
