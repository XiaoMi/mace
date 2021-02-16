add_library(port_windows STATIC
  env.cc
  file_system.cc
)

target_link_libraries(port_windows port_base)
