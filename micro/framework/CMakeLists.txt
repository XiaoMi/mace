add_library(micro_framework
  graph.cc
  micro_engine.cc
  op_context.cc
  operator.cc
  scratch_buffer.cc
)
target_link_libraries(micro_framework micro_include micro_base micro_model)

add_library(micro_framework_for_optest
  graph.cc
  micro_engine.cc
  op_context.cc
  scratch_buffer.cc
)
target_link_libraries(micro_framework_for_optest micro_include micro_base micro_model)
