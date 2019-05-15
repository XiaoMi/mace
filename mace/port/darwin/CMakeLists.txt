add_library(port_darwin STATIC
  env.cc
)

target_link_libraries(port_darwin port_posix)

install(TARGETS port_darwin ARCHIVE DESTINATION lib)
