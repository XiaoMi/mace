add_library(port_posix STATIC
  file_system.cc
)

target_link_libraries(port_posix port_base)

install(TARGETS port_posix ARCHIVE DESTINATION lib)
