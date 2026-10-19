#if defined(_WIN32)
#define ORMI_TEST_EXPORT __declspec(dllexport)
#else
#define ORMI_TEST_EXPORT __attribute__((visibility("default")))
#endif

namespace {
    void write(void* const* outputs, const void* const* inputs,
        unsigned long long begin, unsigned long long end, float increment) {
        auto* output = static_cast<float*>(outputs[0]);
        const auto* input = static_cast<const float*>(inputs[0]);
        for (auto i = begin; i < end; ++i) {
            output[i] = input[i] + increment;
        }
    }
}

extern "C" ORMI_TEST_EXPORT void ormi_kernel(
    void* const* outputs, const void* const* inputs,
    unsigned long long begin, unsigned long long end,
    unsigned long long, const void*) {
    write(outputs, inputs, begin, end, 1.0f);
}

extern "C" ORMI_TEST_EXPORT void ormi_kernel_0(
    void* const* outputs, const void* const* inputs,
    unsigned long long begin, unsigned long long end,
    unsigned long long, const void*) {
    write(outputs, inputs, begin, end, 2.0f);
}

#if !defined(ORMI_TEST_INCOMPLETE_ARTIFACT)
extern "C" ORMI_TEST_EXPORT void ormi_kernel_1(
    void* const* outputs, const void* const* inputs,
    unsigned long long begin, unsigned long long end,
    unsigned long long, const void*) {
    write(outputs, inputs, begin, end, 3.0f);
}
#endif
