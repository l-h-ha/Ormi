#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "core/memory.h"
#include "core/ops.h"

namespace py = pybind11;

namespace ormi {
    void bind_operation(py::module_& m) {
        py::enum_<ormi::ops::OpEnum>(m, "OpEnum")
            .value("LEAF", ormi::ops::OpEnum::LEAF).value("CONST", ormi::ops::OpEnum::CONST)
            .value("ADD", ormi::ops::OpEnum::ADD).value("MUL", ormi::ops::OpEnum::MUL)
            .value("SUB", ormi::ops::OpEnum::SUB).value("DIV", ormi::ops::OpEnum::DIV)
            .value("POW", ormi::ops::OpEnum::POW).value("MATMUL", ormi::ops::OpEnum::MATMUL)
            .value("NEG", ormi::ops::OpEnum::NEG).value("EXP", ormi::ops::OpEnum::EXP)
            .value("LOG", ormi::ops::OpEnum::LOG).value("SIN", ormi::ops::OpEnum::SIN)
            .value("COS", ormi::ops::OpEnum::COS).value("SUM", ormi::ops::OpEnum::SUM)
            .value("TRANSPOSE", ormi::ops::OpEnum::TRANSPOSE)
            .value("RESHAPE", ormi::ops::OpEnum::RESHAPE)
            .value("EXPAND", ormi::ops::OpEnum::EXPAND)
            .export_values();
    }
}