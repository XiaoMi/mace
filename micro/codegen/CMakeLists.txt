file(GLOB_RECURSE generated_models_srcs models *.cc)
file(GLOB_RECURSE micro_engine_srcs engines micro_engine_factory.cc)
file(GLOB_RECURSE micro_engine_c_srcs engines micro_engine_c_interface.cc)

#  Use ".keep.cc" as a source file when there are no model source files in "models" directory
add_library(models
  ${generated_models_srcs}
  ${micro_engine_srcs}
  ${micro_engine_c_srcs}
)
target_link_libraries(models
  micro
)

install(TARGETS models
  ARCHIVE DESTINATION lib
)