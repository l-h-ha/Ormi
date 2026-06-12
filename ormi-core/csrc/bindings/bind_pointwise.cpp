#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/ops/pointwise/add.h"
#include "core/ops/pointwise/sub.h"
#include "core/ops/pointwise/mul.h"
#include "core/ops/pointwise/div.h"
#include "core/ops/pointwise/sin.h"
#include "core/ops/pointwise/cos.h"
#include "core/ops/pointwise/neg.h"
#include "core/ops/pointwise/exp.h"

namespace py = pybind11;

namespace ormi {
    void bind_pointwise(py::module_& m) {
        m.def("add", &ops::add);
        m.def("sub", &ops::sub);
        m.def("mul", &ops::mul);
        m.def("div", &ops::div);
        m.def("sin", &ops::sin);
        m.def("cos", &ops::cos);
        m.def("neg", &ops::neg);
        m.def("exp", &ops::exp);
    }
}