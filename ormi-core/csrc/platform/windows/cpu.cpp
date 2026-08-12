#include <algorithm>
#include <cstddef>
#include <thread>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#if defined(_M_X64) || defined(__x86_64__)
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif
#endif

#include "platform/platform.h"

namespace ormi::platform {
    bool supports_avx2() noexcept {
#if defined(_M_X64) || defined(__x86_64__)
        return IsProcessorFeaturePresent(
            PF_AVX2_INSTRUCTIONS_AVAILABLE
        ) != FALSE;
#else
        return false;
#endif
    }

    bool supports_fma() noexcept {
#if defined(_M_X64) || defined(__x86_64__)
        if (!supports_avx2()) {
            return false;
        }
#if defined(_MSC_VER)
        int registers[4]{};
        __cpuid(registers, 1);
        return (registers[2] & (1 << 12)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
        unsigned int eax = 0;
        unsigned int ebx = 0;
        unsigned int ecx = 0;
        unsigned int edx = 0;
        return __get_cpuid(1, &eax, &ebx, &ecx, &edx) != 0 &&
            (ecx & bit_FMA) != 0;
#else
        return false;
#endif
#else
        return false;
#endif
    }

    bool supports_f16c() noexcept {
#if defined(_M_X64) || defined(__x86_64__)
        if (!supports_avx2()) {
            return false;
        }
#if defined(_MSC_VER)
        int registers[4]{};
        __cpuid(registers, 1);
        return (registers[2] & (1 << 29)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
        unsigned int eax = 0;
        unsigned int ebx = 0;
        unsigned int ecx = 0;
        unsigned int edx = 0;
        return __get_cpuid(1, &eax, &ebx, &ecx, &edx) != 0 &&
            (ecx & bit_F16C) != 0;
#else
        return false;
#endif
#else
        return false;
#endif
    }

    std::size_t cpu_cores() {
        DWORD bytes = 0;
        GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            nullptr,
            &bytes
        );
        if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || bytes == 0) {
            return std::max<std::size_t>(
                std::thread::hardware_concurrency(),
                1
            );
        }

        std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX> storage(
            (bytes + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) - 1) /
                sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)
        );
        if (!GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
                storage.data()
            ),
            &bytes
        )) {
            return std::max<std::size_t>(
                std::thread::hardware_concurrency(),
                1
            );
        }

        std::size_t cores = 0;
        std::size_t offset = 0;
        while (offset < bytes) {
            const auto* entry = reinterpret_cast<
                const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*
            >(
                reinterpret_cast<const unsigned char*>(storage.data()) +
                    offset
            );
            if (entry->Relationship == RelationProcessorCore) {
                ++cores;
            }
            if (entry->Size == 0) {
                break;
            }
            offset += entry->Size;
        }
        return std::max<std::size_t>(cores, 1);
    }
}
