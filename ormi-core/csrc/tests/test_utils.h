#pragma once

#include <gtest/gtest.h>
#include <vector>

#include "core/device.h"
#include "core/allocator.h"

namespace ormi::testing {
    class HardwareTestSuite : public ::testing::Test {
    protected:
        std::vector<Device> available_devices;

        void SetUp() override {
            available_devices.push_back(Device::cpu());

            try {
                IAllocator* cuda_alloc = get_allocator(Device::cuda(0));
                if (cuda_alloc != nullptr) {
                    available_devices.push_back(Device::cuda(0));
                }
            } catch (...) {
                // CUDA is unavailable on this machine.
            }
        }
    };
}