#pragma once
#include <stdexcept>
#include <string>

namespace ormi {
    class OrmiError : public std::runtime_error {
    public:
        explicit OrmiError(const std::string& msg) : std::runtime_error(msg) {}
    };

    class DeviceError : public OrmiError { using OrmiError::OrmiError; };
    class DTypeError : public OrmiError { using OrmiError::OrmiError; };
    class ShapeError : public OrmiError { using OrmiError::OrmiError; };
    class OperationError : public OrmiError { using OrmiError::OrmiError; };
    class BufferError : public OrmiError { using OrmiError::OrmiError; };
    class MemoryError : public OrmiError { using OrmiError::OrmiError; };
    class BackendError : public DeviceError { using DeviceError::DeviceError; };

    class jitError : public OrmiError { using OrmiError::OrmiError; };

    namespace detail {
        template <typename ExceptionType>
        [[noreturn]] inline void throw_error(const std::string& message) {
            throw ExceptionType(message);
        }
    }
}

#define ORMI_THROW(ExceptionType, msg) \
    ormi::detail::throw_error<ExceptionType>(msg)
