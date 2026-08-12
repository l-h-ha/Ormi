ormi_platform(
    NAME windows
    SYSTEM Windows
    SOURCES
        platform/windows/utf.cpp
        platform/windows/process.cpp
        platform/windows/toolchain.cpp
        platform/windows/library.cpp
        platform/windows/filesystem.cpp
        platform/windows/cpu.cpp
    EXPORT_ALL
    NO_PREFIX
    STAGE
)
