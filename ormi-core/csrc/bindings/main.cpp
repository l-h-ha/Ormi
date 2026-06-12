#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "core/autograd.h"
#include "backend/runtime.h"

namespace py = pybind11;

namespace ormi {
    void bind_blas(py::module_& m);
    void bind_device(py::module_& m);
    void bind_graph(py::module_& m);
    void bind_memory(py::module_& m);
    void bind_operation(py::module_& m);
    void bind_pointwise(py::module_& m);
    void bind_reduce(py::module_& m);
    void bind_view(py::module_& m);
}

PYBIND11_MODULE(ormi_backend, m) {
    m.doc() = "Ormi C++ Backend";

    ormi::bind_device(m);
    ormi::bind_operation(m);

    ormi::bind_memory(m);
    ormi::bind_graph(m);

    ormi::bind_blas(m);
    ormi::bind_pointwise(m);
    ormi::bind_reduce(m);
    ormi::bind_view(m);

    m.def("execute_graph", &ormi::execute_graph);
    m.def("compute_gradients", &ormi::compute_gradients);
    m.def("zero_grad_graph", &ormi::zero_grad_graph);
}