add_library(port_qnx STATIC
  env.cc
)

target_link_libraries(port_qnx port_posix)