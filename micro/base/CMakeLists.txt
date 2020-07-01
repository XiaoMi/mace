add_library(micro_base
  logger.cc
  serialize.cc
  utils.cc
  value_to_str.cc
)
target_link_libraries(micro_base
  PUBLIC micro_port
  PUBLIC micro_include
)
target_compile_options(micro_base PUBLIC -fPIC)
