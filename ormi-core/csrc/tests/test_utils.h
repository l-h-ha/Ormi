#pragma once
#include <gtest/gtest.h>
#include <vector>
#include "core/device.h"
#include "core/allocator.h"

namespace ormi::testing {
    class HardwareTestSuite : public ::testing::Test {
    protected:
        std::vector<DeviceEnum> available_devices;

        void SetUp() override {
            // CPU
            available_devices.push_back(DeviceEnum::CPU);
            
            // CUDA
            try {
                if (get_allocator(DeviceEnum::CUDA) != nullptr) {
                    available_devices.push_back(DeviceEnum::CUDA);
                }
            } catch (...) {}
        }
    };
}