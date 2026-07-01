#pragma once
#include <string>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <iostream>

namespace ormi {
    class OrmiError : public std::runtime_error { public: explicit OrmiError(const std::string& msg) : std::runtime_error(msg) {} };
    class DeviceError : public OrmiError { using OrmiError::OrmiError; };
    class DTypeError : public OrmiError { using OrmiError::OrmiError; };
    class ShapeError : public OrmiError { using OrmiError::OrmiError; };
    class RegistryError : public OrmiError { using OrmiError::OrmiError; };
    class OperationError : public OrmiError { using OrmiError::OrmiError; };
    class BufferError : public OrmiError { using OrmiError::OrmiError; };
    class NodeError : public OrmiError { using OrmiError::OrmiError; };
    class MemoryError : public OrmiError { using OrmiError::OrmiError; };
    class ValueError : public OrmiError { using OrmiError::OrmiError; };
    class AutogradError : public OrmiError { using OrmiError::OrmiError; };
    class RuntimeError : public OrmiError { using OrmiError::OrmiError; };

    namespace detail {
        template <typename T>
        inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
            os << "[";
            for (size_t i = 0; i < vec.size(); ++i) {
                os << vec[i] << (i == vec.size() - 1 ? "" : ", ");
            }
            return os << "]";
        }

        template <typename ExceptionType = OrmiError>
        [[noreturn]] inline void throw_error(const char* file, int line, const std::string& msg) {
            std::ostringstream oss;
            oss << "[ORMI ERROR] " << file << ":" << line << "\n" << msg;
            throw ExceptionType(oss.str());
        }

        template <typename ExceptionType = OrmiError, typename T1, typename T2>
        [[noreturn]] inline void throw_expected_got(const char* file, int line, const T1& expected, const T2& got, const std::string& msg) {
            std::ostringstream oss;
            oss << "[ORMI ERROR] " << file << ":" << line << "\n";
            if (!msg.empty()) oss << msg << "\n";
            oss << "  Expected: " << expected << "\n";
            oss << "  Got:      " << got;
            throw ExceptionType(oss.str());
        }

        inline void print_warning(const char* file, int line, const std::string& msg) {
            std::cerr << "[ORMI WARNING] " << file << ":" << line << "\n" << msg << "\n";
        }
    }
}

#define ORMI_WARN(msg) \
    ormi::detail::print_warning(__FILE__, __LINE__, msg)

#define ORMI_FAIL(msg) \
    ormi::detail::throw_error<ormi::OrmiError>(__FILE__, __LINE__, msg)

#define ORMI_THROW(ExceptionType, msg) \
    ormi::detail::throw_error<ExceptionType>(__FILE__, __LINE__, msg)

#define ORMI_CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            ormi::detail::throw_error<ormi::OrmiError>(__FILE__, __LINE__, msg); \
        } \
    } while (false)

#define ORMI_CHECK_EX(cond, ExceptionType, msg) \
    do { \
        if (!(cond)) { \
            ormi::detail::throw_error<ExceptionType>(__FILE__, __LINE__, msg); \
        } \
    } while (false)

#define ORMI_CHECK_EQ(got, expected, ExceptionType, msg) \
    do { \
        if (!((got) == (expected))) { \
            ormi::detail::throw_expected_got<ExceptionType>(__FILE__, __LINE__, expected, got, msg); \
        } \
    } while (false)
