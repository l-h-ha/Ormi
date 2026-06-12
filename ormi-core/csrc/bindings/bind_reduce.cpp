#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "core/ops/reduce/sum.h"

namespace py = pybind11;

namespace ormi {
    void bind_reduce(py::module_& m) {
        m.def("sum", &ormi::ops::sum);
    }
}