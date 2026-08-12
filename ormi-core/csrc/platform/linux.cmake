ormi_platform(
    NAME linux
    SYSTEM Linux
    SOURCES platform/linux.cpp
    PLUGIN_RPATH "$ORIGIN"
    LINKS ${CMAKE_DL_LIBS}
)
