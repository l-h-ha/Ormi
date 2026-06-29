#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

namespace ormi::bindings {
    void bind_generated(pybind11::module_& m);
}

PYBIND11_MODULE(_C, m) {
    m.doc() = "Ormi C++ Backend";
    ormi::bindings::bind_generated(m);
}