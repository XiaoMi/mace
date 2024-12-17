add_library(port_base STATIC
  env.cc
  logger.cc
  file_system.cc
)

target_link_libraries(port_base utils)

if(ANDROID)
  add_subdirectory(posix)
  add_subdirectory(linux_base)
  add_subdirectory(android)
  add_library(port ALIAS port_android)
elseif(APPLE)
  add_subdirectory(posix)
  add_subdirectory(darwin)
  add_library(port ALIAS port_darwin)
elseif(WIN32)
  add_subdirectory(windows)
  add_library(port ALIAS port_windows)
elseif(QNX)
  add_subdirectory(posix)
  add_subdirectory(qnx)
  add_library(port ALIAS port_qnx)
else(WIN32)
  add_subdirectory(posix)
  add_subdirectory(linux_base)
  add_subdirectory(linux)
  add_library(port ALIAS port_linux)
endif(ANDROID)
