#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "core/ops/view/transpose.h"
#include "core/ops/view/expand.h"
#include "core/ops/view/reshape.h"

namespace py = pybind11;

namespace ormi {
    void bind_view(py::module_& m) {
        m.def("transpose", &ormi::ops::transpose);
        m.def("expand", &ormi::ops::expand);
        m.def("reshape", &ormi::ops::reshape);
    }
}