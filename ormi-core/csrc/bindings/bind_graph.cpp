#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "core/shape.h"
#include "core/memory.h"
#include "core/node.h"

namespace py = pybind11;

namespace ormi {
    void bind_graph(py::module_& m) {
        py::class_<ormi::TensorView>(m, "TensorView")
            .def_readwrite("shape", &ormi::TensorView::shape)
            .def_readwrite("strides", &ormi::TensorView::strides)
            .def("is_contiguous", &ormi::TensorView::is_contiguous);

        py::class_<ormi::Node, std::shared_ptr<ormi::Node>>(m, "Node")
            .def(py::init<std::shared_ptr<ormi::Buffer>, bool, ormi::Device>())
            .def(py::init<ormi::ops::OpEnum, std::vector<std::shared_ptr<ormi::Node>>, ormi::Device>())
            .def_property_readonly("op", [](const ormi::Node& n) { return n.op; })
            .def_property_readonly("requires_grad", [](const ormi::Node& n) { return n.requires_grad; })
            .def_property_readonly("is_realized", &ormi::Node::is_realized)
            .def_property_readonly("shape", [](const ormi::Node& n) { return n.view.shape; })
            .def_property_readonly("strides", [](const ormi::Node& n) { return n.view.strides; })
            .def_readwrite("view", &ormi::Node::view)
            .def_readwrite("data", &ormi::Node::data)
            .def_readwrite("grad", &ormi::Node::grad);
    }
}