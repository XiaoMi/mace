add_library(micro_model
  argument.cc
  const_tensor.cc
  input_output_info.cc
  net_def.cc
  operator_def.cc
  output_shape.cc
)

target_link_libraries(micro_model micro_base micro_include)
