add_library(micro_framework
  graph.cc
  micro_engine.cc
  op_context.cc
  operator.cc
  scratch_buffer.cc
)
target_link_libraries(micro_framework
  PUBLIC micro_include
  PUBLIC micro_base
  PUBLIC micro_model
)

add_library(micro_framework_for_optest
  graph.cc
  micro_engine.cc
  op_context.cc
  scratch_buffer.cc
)
target_link_libraries(micro_framework_for_optest
  PUBLIC micro_include
  PUBLIC micro_base
  PUBLIC micro_model
)
