#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "core/memory.h"

namespace py = pybind11;

namespace ormi {
    void bind_device(py::module_& m) {
        py::enum_<ormi::DeviceEnum>(m, "DeviceEnum")
            .value("CPU", ormi::DeviceEnum::CPU)
            .value("CUDA", ormi::DeviceEnum::CUDA)
            .export_values();

        py::class_<ormi::Device>(m, "Device")
            .def(
                py::init<ormi::DeviceEnum, int>(), 
                py::arg_v("type", ormi::DeviceEnum::CUDA, "DeviceEnum.CUDA"), 
                py::arg("id") = 0
            )
            .def_readwrite("type", &ormi::Device::type)
            .def_readwrite("id", &ormi::Device::id)
            .def("__repr__", &ormi::Device::to_string)
            .def("is_type", [](const ormi::Device& d, const std::string& name) {
                if (name == "cuda" || name == "CUDA") return d.type == ormi::DeviceEnum::CUDA;
                if (name == "cpu" || name == "CPU") return d.type == ormi::DeviceEnum::CPU;
                return false;
            });
    }
}