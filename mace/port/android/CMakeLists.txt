add_library(port_android STATIC
  logger.cc
  malloc_logger.cc
  env.cc
)

target_link_libraries(port_android port_linux_base log)

install(TARGETS port_android ARCHIVE DESTINATION lib)
