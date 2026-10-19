ormi_platform(
    NAME linux
    SYSTEM Linux
    SOURCES
        platform/linux.cpp
        platform/linux/library.cpp
        platform/linux/process.cpp
    PLUGIN_RPATH "$ORIGIN"
    LINKS ${CMAKE_DL_LIBS}
)
