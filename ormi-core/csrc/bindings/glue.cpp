#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ormi/core/attrs.h>
#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/memory/ops.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "core/runtime/dispatch/device.h"
#include "bindings/glue.h"

namespace ormi::bindings {
    namespace {
        py::dtype numpy_dtype(core::DType dtype) {
            switch (dtype) {
                case core::DType::float64:
                    return py::dtype::of<double>();
                case core::DType::float32:
                    return py::dtype::of<float>();
                case core::DType::float16:
                    return py::dtype("float16");
                case core::DType::int32:
                    return py::dtype::of<std::int32_t>();
                case core::DType::boolean:
                    return py::dtype::of<bool>();
            }

            ORMI_THROW(DTypeError, "Cannot map an unknown dtype to NumPy.");
        }

        core::Scalar scalar(py::handle value, core::DType dtype) {
            switch (dtype) {
                case core::DType::float64:
                    return value.cast<double>();
                case core::DType::float32:
                    return value.cast<float>();
                case core::DType::float16:
                    return core::Float16(value.cast<float>());
                case core::DType::int32:
                    return value.cast<std::int32_t>();
                case core::DType::boolean:
                    return value.cast<bool>();
            }

            ORMI_THROW(DTypeError, "Cannot convert a Python scalar to an unknown dtype.");
        }

        core::Device as_device(py::handle value) {
            if (value.is_none()) {
                return core::Device::cpu();
            }
            if (py::isinstance<core::Device>(value)) {
                return value.cast<core::Device>();
            }
            if (py::isinstance<py::str>(value)) {
                return core::Device::from(value.cast<std::string>());
            }
            throw py::type_error("device must be an ORMI Device, string, or None");
        }

        std::shared_ptr<core::Tensor> as_tensor(
            py::handle value,
            const std::shared_ptr<core::Tensor>& like
        ) {
            if (py::isinstance<core::Tensor>(value)) {
                return value.cast<std::shared_ptr<core::Tensor>>();
            }

            return core::ops::full(
                core::Shape{},
                scalar(value, like->dtype()),
                like->device()
            );
        }

        template <typename Fn>
        std::shared_ptr<core::Tensor> binary(
            const std::shared_ptr<core::Tensor>& lhs,
            py::handle rhs,
            Fn function
        ) {
            if (!lhs) {
                ORMI_THROW(OperationError, "A Python tensor operation received a null tensor.");
            }
            return function(lhs, as_tensor(rhs, lhs));
        }

        template <typename Fn>
        std::shared_ptr<core::Tensor> reversed_binary(
            py::handle lhs,
            const std::shared_ptr<core::Tensor>& rhs,
            Fn function
        ) {
            if (!rhs) {
                ORMI_THROW(OperationError, "A Python tensor operation received a null tensor.");
            }
            return function(as_tensor(lhs, rhs), rhs);
        }

        std::vector<py::ssize_t> numpy_shape(const core::Shape& shape) {
            std::vector<py::ssize_t> result;
            result.reserve(shape.rank());
            for (const auto dim : shape.dims()) {
                result.push_back(static_cast<py::ssize_t>(dim));
            }
            return result;
        }

        void copy_strided_to_contiguous(
            unsigned char* dst,
            const unsigned char* src,
            const core::Tensor& tensor
        ) {
            const auto item_size = core::dtype_size(tensor.dtype());
            const auto strides = tensor.strides();
            const auto& shape = tensor.shape();

            for (std::size_t linear = 0; linear < tensor.numel(); ++linear) {
                auto remaining = linear;
                std::size_t source_index = 0;

                for (std::size_t axis = shape.rank(); axis-- > 0;) {
                    const auto extent = static_cast<std::size_t>(shape[axis]);
                    const auto index = extent == 0 ? 0 : remaining % extent;
                    remaining = extent == 0 ? 0 : remaining / extent;
                    source_index += index * static_cast<std::size_t>(strides[axis]);
                }

                std::memcpy(
                    dst + linear * item_size,
                    src + source_index * item_size,
                    item_size
                );
            }
        }
    }

    core::Device device_from_string(const std::string& value) {
        return core::Device::from(value);
    }

    std::string device_string(core::Device device) {
        std::string value(device.name());
        if (device.indexed()) {
            value += ":" + std::to_string(device.index());
        }
        return value;
    }

    std::string device_repr(core::Device device) {
        return "Device('" + device_string(device) + "')";
    }

    bool device_equal(core::Device lhs, core::Device rhs) {
        return lhs.backend_id() == rhs.backend_id() && lhs.index() == rhs.index();
    }

    std::shared_ptr<core::Tensor> tensor_from_python(
        py::object data,
        core::DType dtype,
        py::object device_value
    ) {
        const auto device = as_device(device_value);
        auto numpy = py::module_::import("numpy");
        auto array = numpy.attr("asarray")(data, numpy_dtype(dtype)).cast<py::array>();
        if ((array.flags() & py::array::c_style) == 0) {
            array = numpy.attr("ascontiguousarray")(array).cast<py::array>();
        }
        const auto info = array.request();

        std::vector<std::int64_t> dimensions;
        dimensions.reserve(static_cast<std::size_t>(info.ndim));
        for (const auto dim : info.shape) {
            dimensions.push_back(static_cast<std::int64_t>(dim));
        }

        auto result = std::make_shared<core::Tensor>(
            device,
            core::Shape(std::move(dimensions)),
            dtype
        );

        const auto bytes = result->bytes();
        if (bytes != static_cast<std::size_t>(info.size * info.itemsize)) {
            ORMI_THROW(BufferError, "NumPy input size does not match the requested ORMI dtype.");
        }

        if (bytes != 0) {
            py::gil_scoped_release release;
            core::memory::copy(result->data(), device, info.ptr, core::Device::cpu(), bytes);
        }
        return result;
    }

    std::unique_ptr<core::Plan> plan_from_python(
        std::vector<std::shared_ptr<core::Tensor>> outputs,
        std::vector<std::pair<
            std::shared_ptr<core::Tensor>,
            std::shared_ptr<core::Tensor>
        >> updates
    ) {
        std::vector<core::Plan::Update> converted;
        converted.reserve(updates.size());
        for (auto& [destination, value] : updates) {
            converted.push_back(core::Plan::Update{
                std::move(destination),
                std::move(value)
            });
        }
        return std::make_unique<core::Plan>(
            std::move(outputs),
            std::move(converted)
        );
    }

    py::array tensor_numpy(const std::shared_ptr<core::Tensor>& tensor) {
        if (!tensor) {
            ORMI_THROW(OperationError, "Cannot convert a null tensor to NumPy.");
        }

        py::array result(numpy_dtype(tensor->dtype()), numpy_shape(tensor->shape()));
        if (tensor->numel() == 0) {
            return result;
        }

        auto* destination = static_cast<unsigned char*>(result.mutable_data());
        py::gil_scoped_release release;

        if (tensor->is_contiguous()) {
            core::memory::copy(
                destination,
                core::Device::cpu(),
                tensor->data(),
                tensor->device(),
                tensor->bytes()
            );
            return result;
        }

        const auto buffer = tensor->buffer();
        std::vector<unsigned char> host(buffer->bytes());
        core::memory::copy(
            host.data(),
            core::Device::cpu(),
            buffer->data(),
            tensor->device(),
            buffer->bytes()
        );
        copy_strided_to_contiguous(
            destination,
            host.data() + tensor->offset_bytes(),
            *tensor
        );
        return result;
    }

    py::object tensor_item(const std::shared_ptr<core::Tensor>& tensor) {
        if (!tensor || tensor->numel() != 1) {
            ORMI_THROW(OperationError, "Tensor.item() requires exactly one element.");
        }
        return tensor_numpy(tensor).attr("item")();
    }

    std::shared_ptr<core::Tensor> tensor_realize(const std::shared_ptr<core::Tensor>& tensor) {
        if (!tensor) {
            ORMI_THROW(OperationError, "Cannot realize a null tensor.");
        }
        py::gil_scoped_release release;
        tensor->realize();
        return tensor;
    }

    py::tuple tensor_shape(const core::Tensor& tensor) {
        py::tuple result(tensor.rank());
        for (std::size_t axis = 0; axis < tensor.rank(); ++axis) {
            result[axis] = tensor.shape()[axis];
        }
        return result;
    }

    py::tuple tensor_strides(const core::Tensor& tensor) {
        const auto strides = tensor.strides();
        py::tuple result(strides.size());
        for (std::size_t axis = 0; axis < strides.size(); ++axis) {
            result[axis] = strides[axis];
        }
        return result;
    }

    std::string tensor_op(const core::Tensor& tensor) {
        return tensor.op() ? std::string(tensor.op()->name) : "leaf";
    }

    std::string tensor_repr(const core::Tensor& tensor) {
        std::ostringstream output;
        output << "Tensor(shape=" << tensor.shape()
               << ", dtype=" << tensor.dtype()
               << ", device='" << device_string(tensor.device())
               << "', op='" << tensor_op(tensor)
               << "', " << (tensor.realized() ? "realized" : "lazy") << ")";
        return output.str();
    }

    std::shared_ptr<core::Tensor> tensor_add(
        const std::shared_ptr<core::Tensor>& lhs,
        py::object rhs
    ) {
        return binary(lhs, rhs, core::ops::add);
    }

    std::shared_ptr<core::Tensor> tensor_radd(
        const std::shared_ptr<core::Tensor>& rhs,
        py::object lhs
    ) {
        return reversed_binary(lhs, rhs, core::ops::add);
    }

    std::shared_ptr<core::Tensor> tensor_sub(
        const std::shared_ptr<core::Tensor>& lhs,
        py::object rhs
    ) {
        return binary(lhs, rhs, core::ops::sub);
    }

    std::shared_ptr<core::Tensor> tensor_rsub(
        const std::shared_ptr<core::Tensor>& rhs,
        py::object lhs
    ) {
        return reversed_binary(lhs, rhs, core::ops::sub);
    }

    std::shared_ptr<core::Tensor> tensor_mul(
        const std::shared_ptr<core::Tensor>& lhs,
        py::object rhs
    ) {
        return binary(lhs, rhs, core::ops::mul);
    }

    std::shared_ptr<core::Tensor> tensor_rmul(
        const std::shared_ptr<core::Tensor>& rhs,
        py::object lhs
    ) {
        return reversed_binary(lhs, rhs, core::ops::mul);
    }

    std::shared_ptr<core::Tensor> tensor_div(
        const std::shared_ptr<core::Tensor>& lhs,
        py::object rhs
    ) {
        return binary(lhs, rhs, core::ops::div);
    }

    std::shared_ptr<core::Tensor> tensor_rdiv(
        const std::shared_ptr<core::Tensor>& rhs,
        py::object lhs
    ) {
        return reversed_binary(lhs, rhs, core::ops::div);
    }

    std::shared_ptr<core::Tensor> tensor_sum(
        const std::shared_ptr<core::Tensor>& input,
        py::object axes,
        bool keepdims
    ) {
        if (axes.is_none()) {
            return core::ops::sum(input, keepdims);
        }
        return core::ops::sum(input, axes.cast<std::vector<std::int64_t>>(), keepdims);
    }

    std::shared_ptr<core::Tensor> tensor_mean(
        const std::shared_ptr<core::Tensor>& input,
        py::object axes,
        bool keepdims
    ) {
        if (axes.is_none()) {
            return core::ops::mean(input, keepdims);
        }
        return core::ops::mean(input, axes.cast<std::vector<std::int64_t>>(), keepdims);
    }

    std::shared_ptr<core::Tensor> tensor_transpose(
        const std::shared_ptr<core::Tensor>& input,
        py::object axes
    ) {
        std::vector<std::int64_t> permutation;
        if (axes.is_none()) {
            permutation.reserve(input->rank());
            for (std::size_t axis = input->rank(); axis-- > 0;) {
                permutation.push_back(static_cast<std::int64_t>(axis));
            }
        } else {
            permutation = axes.cast<std::vector<std::int64_t>>();
        }
        return core::ops::transpose(input, std::move(permutation));
    }

    std::shared_ptr<core::Tensor> tensor_reshape(
        const std::shared_ptr<core::Tensor>& input,
        std::vector<std::int64_t> shape
    ) {
        return core::ops::reshape(input, core::Shape(std::move(shape)));
    }

    std::shared_ptr<core::Tensor> tensor_expand(
        const std::shared_ptr<core::Tensor>& input,
        std::vector<std::int64_t> shape
    ) {
        return core::ops::expand(input, core::Shape(std::move(shape)));
    }

    std::shared_ptr<core::Tensor> full(
        std::vector<std::int64_t> shape,
        py::object value,
        core::DType dtype,
        py::object device_value
    ) {
        const auto device = as_device(device_value);
        return core::ops::full(core::Shape(std::move(shape)), scalar(value, dtype), device);
    }

    std::shared_ptr<core::Tensor> zeros(
        std::vector<std::int64_t> shape,
        core::DType dtype,
        py::object device
    ) {
        return full(std::move(shape), py::int_(0), dtype, device);
    }

    std::shared_ptr<core::Tensor> ones(
        std::vector<std::int64_t> shape,
        core::DType dtype,
        py::object device
    ) {
        return full(std::move(shape), py::int_(1), dtype, device);
    }

    void synchronize(core::Device device) {
        py::gil_scoped_release release;
        core::runtime::device::synchronize(device);
    }
}
