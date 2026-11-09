#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <ormi/core/autograd.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/index.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/view.h>

namespace ormi::core {
    namespace {
        std::int32_t selector(std::optional<std::int64_t> value, std::int64_t size, const char* name) {
            if (!value && size != 1) {
                ORMI_THROW(
                    ShapeError,
                    std::string("Partial requires an explicit ") + name + " when its dimension is not one."
                );
            }
            auto index = value.value_or(0);
            if (index < 0) {
                index += size;
            }
            if (index < 0 || index >= size) {
                ORMI_THROW(ShapeError, std::string("Partial ") + name + " is out of range.");
            }
            if (index > std::numeric_limits<std::int32_t>::max()) {
                ORMI_THROW(ShapeError, std::string("Partial ") + name + " exceeds the Int32 indexing limit.");
            }
            return static_cast<std::int32_t>(index);
        }
    }

    std::shared_ptr<Tensor> partial(
        const std::shared_ptr<Tensor>& output,
        const std::shared_ptr<Tensor>& input,
        std::optional<std::int64_t> coordinate,
        std::optional<std::int64_t> component,
        std::size_t order
    ) {
        if (!output || !input) {
            ORMI_THROW(OperationError, "Partial differentiation requires output and input tensors.");
        }
        if (order == 0) {
            ORMI_THROW(OperationError, "Partial differentiation order must be positive.");
        }
        if (!is_floating(output->dtype()) || !is_floating(input->dtype())) {
            ORMI_THROW(DTypeError, "Partial differentiation supports floating-point tensors only.");
        }
        if (output->device() != input->device()) {
            ORMI_THROW(DeviceError, "Partial output and input must use the same device.");
        }
        if (output->rank() == 0 || input->rank() == 0) {
            ORMI_THROW(
                ShapeError,
                "Partial requires trailing output and coordinate dimensions. Use grad for scalar tensors."
            );
        }

        auto points = input->shape().dims();
        const auto coordinates = points.back();
        points.pop_back();
        auto output_points = output->shape().dims();
        const auto components = output_points.back();
        output_points.pop_back();
        if (points != output_points) {
            ORMI_THROW(ShapeError, "Partial output and input must have matching point dimensions.");
        }
        const auto input_channel = selector(coordinate, coordinates, "coordinate");
        const auto output_channel = selector(component, components, "component");
        for (const auto size : points) {
            if (size - 1 > std::numeric_limits<std::int32_t>::max()) {
                ORMI_THROW(ShapeError, "Partial point dimensions exceed the Int32 indexing limit.");
            }
        }

        const auto indices = ops::full(Shape(points), input_channel, input->device());
        auto result = ops::zeros_like(ops::pick(input, indices));
        std::vector<std::int64_t> point(points.size(), 0);
        const auto count = input->numel() / static_cast<std::size_t>(coordinates);

        for (std::size_t current = 0; current < count; ++current) {
            const auto output_index = current * static_cast<std::size_t>(components) + output_channel;
            const auto input_index = current * static_cast<std::size_t>(coordinates) + input_channel;
            auto value = ops::select(output, output_index);
            // Every reverse pass starts from one scalar. No output sums or point-independence assumptions.
            for (std::size_t derivative = 0; derivative < order; ++derivative) {
                value = ops::select(grad(value, input), input_index);
            }

            std::vector<std::int64_t> dimensions;
            for (std::size_t axis = 0; axis < points.size(); ++axis) {
                const auto position =
                    ops::full(value->shape(), static_cast<std::int32_t>(point[axis]), input->device());
                dimensions.push_back(points[axis]);
                value = ops::unpick(value, position, Shape(dimensions));
            }
            result = ops::add(result, value);

            for (std::size_t axis = points.size(); axis > 0; --axis) {
                if (++point[axis - 1] < points[axis - 1]) {
                    break;
                }
                point[axis - 1] = 0;
            }
        }

        points.push_back(1);
        return ops::reshape(result, Shape(points));
    }
}
