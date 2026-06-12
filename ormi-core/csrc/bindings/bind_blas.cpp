#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "core/ops/blas/matmul.h"

namespace py = pybind11;

namespace ormi {
    void bind_blas(py::module_& m) {
        m.def("matmul", &ormi::ops::matmul);
    }
}