file(GLOB_RECURSE generated_models_srcs models *.cc)
add_library(generated_models
  ${generated_models_srcs}
)
target_link_libraries(generated_models
  micro_framework
  micro_include
  micro_model
  micro_ops
)

file(GLOB_RECURSE micro_engine_srcs engines micro_engine_factory.cc)
add_library(micro_engine
  ${micro_engine_srcs}
)
target_link_libraries(micro_engine
  micro_framework
  micro_model
  micro_ops
  generated_models
)

file(GLOB_RECURSE micro_engine_c_srcs engines micro_engine_c_interface.cc)
add_library(micro_engine_c
  ${micro_engine_c_srcs}
)
target_link_libraries(micro_engine_c
  micro_engine
)
