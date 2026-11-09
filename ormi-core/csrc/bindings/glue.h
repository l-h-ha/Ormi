#pragma once
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/plan.h>
#include <ormi/core/tensor.h>
#include <ormi/nn/modules/linear.h>
#include <ormi/nn/modules/normalization.h>
#include <ormi/nn/modules/batch_norm.h>

namespace ormi::bindings {
    namespace py = pybind11;

    std::shared_ptr<nn::Linear> linear_from_python(
        std::int64_t in_features, std::int64_t out_features, bool bias,
        core::DType dtype, py::object device, std::optional<std::uint64_t> seed
    );

    std::shared_ptr<nn::LayerNorm> layer_norm_from_python(
        py::object normalized_shape,
        double eps,
        bool elementwise_affine,
        bool bias,
        core::DType dtype,
        py::object device
    );
    std::shared_ptr<core::Tensor> layer_norm(
        const std::shared_ptr<core::Tensor>& input,
        py::object normalized_shape,
        const std::shared_ptr<core::Tensor>& weight,
        const std::shared_ptr<core::Tensor>& bias,
        double eps
    );
    py::tuple layer_norm_shape(const nn::LayerNorm& layer);
    std::shared_ptr<nn::BatchNorm> batch_norm_from_python(
        std::int64_t num_features, double eps, double momentum, bool affine, bool track_running_stats,
        std::int64_t axis, core::DType dtype, py::object device
    );

    core::Device device_from_string(const std::string& value);
    std::string device_string(core::Device device);
    std::string device_repr(core::Device device);
    bool device_equal(core::Device lhs, core::Device rhs);

    std::shared_ptr<core::Tensor> tensor_from_python(
        py::object data,
        core::DType dtype,
        py::object device
    );
    std::unique_ptr<core::Plan> plan_from_python(
        std::vector<std::shared_ptr<core::Tensor>> outputs,
        std::vector<std::pair<
            std::shared_ptr<core::Tensor>,
            std::shared_ptr<core::Tensor>
        >> updates
    );
    py::array tensor_numpy(const std::shared_ptr<core::Tensor>& tensor);
    void tensor_copy_from(core::Tensor& destination, py::object source);
    py::object tensor_item(const std::shared_ptr<core::Tensor>& tensor);
    std::shared_ptr<core::Tensor> tensor_realize(const std::shared_ptr<core::Tensor>& tensor);
    py::tuple tensor_shape(const core::Tensor& tensor);
    py::tuple tensor_strides(const core::Tensor& tensor);
    std::string tensor_op(const core::Tensor& tensor);
    std::string tensor_repr(const core::Tensor& tensor);

    std::shared_ptr<core::Tensor> tensor_add(
        const std::shared_ptr<core::Tensor>& lhs,
        py::object rhs
    );
    std::shared_ptr<core::Tensor> tensor_radd(
        const std::shared_ptr<core::Tensor>& rhs,
        py::object lhs
    );
    std::shared_ptr<core::Tensor> tensor_sub(
        const std::shared_ptr<core::Tensor>& lhs,
        py::object rhs
    );
    std::shared_ptr<core::Tensor> tensor_rsub(
        const std::shared_ptr<core::Tensor>& rhs,
        py::object lhs
    );
    std::shared_ptr<core::Tensor> tensor_mul(
        const std::shared_ptr<core::Tensor>& lhs,
        py::object rhs
    );
    std::shared_ptr<core::Tensor> tensor_rmul(
        const std::shared_ptr<core::Tensor>& rhs,
        py::object lhs
    );
    std::shared_ptr<core::Tensor> tensor_div(
        const std::shared_ptr<core::Tensor>& lhs,
        py::object rhs
    );
    std::shared_ptr<core::Tensor> tensor_rdiv(
        const std::shared_ptr<core::Tensor>& rhs,
        py::object lhs
    );

    std::shared_ptr<core::Tensor> tensor_sum(
        const std::shared_ptr<core::Tensor>& input,
        py::object axes,
        bool keepdims
    );
    std::shared_ptr<core::Tensor> tensor_mean(
        const std::shared_ptr<core::Tensor>& input,
        py::object axes,
        bool keepdims
    );
    std::shared_ptr<core::Tensor> tensor_transpose(
        const std::shared_ptr<core::Tensor>& input,
        py::object axes
    );
    std::shared_ptr<core::Tensor> tensor_reshape(
        const std::shared_ptr<core::Tensor>& input,
        std::vector<std::int64_t> shape
    );
    std::shared_ptr<core::Tensor> tensor_expand(
        const std::shared_ptr<core::Tensor>& input,
        std::vector<std::int64_t> shape
    );

    std::shared_ptr<core::Tensor> full(
        std::vector<std::int64_t> shape,
        py::object value,
        core::DType dtype,
        py::object device
    );
    std::shared_ptr<core::Tensor> zeros(
        std::vector<std::int64_t> shape,
        core::DType dtype,
        py::object device
    );
    std::shared_ptr<core::Tensor> ones(
        std::vector<std::int64_t> shape,
        core::DType dtype,
        py::object device
    );

    void synchronize(core::Device device);
}
