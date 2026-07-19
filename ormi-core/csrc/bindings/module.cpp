#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ormi::bindings {
    void bind_generated(py::module_& module);
}

PYBIND11_MODULE(_C, module) {
    module.doc() = "ORMI C++ lazy tensor runtime";
    ormi::bindings::bind_generated(module);
}
