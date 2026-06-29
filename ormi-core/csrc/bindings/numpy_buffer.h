#pragma once

#include <memory>

#include <pybind11/numpy.h>

#include "core/buffer.h"
#include "core/device.h"

namespace ormi {
    std::shared_ptr<Buffer> numpy_to_buffer(
        pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> array,
        Device device
    );

    pybind11::array_t<float> buffer_to_numpy(const std::shared_ptr<Buffer>& buffer);
}