add_library(mace_micro_ops_nhwc_cmsis_nn
  arm_conv_2d_int8.cc
  arm_pooling_int8.cc
  arm_softmax_int8.cc
  arm_mat_mul_int8.cc
  arm_depthwise_conv_2d_int8.cc
  dequantize.cc
  quantize.cc
  utilities.cc
)

target_link_libraries(mace_micro_ops_nhwc_cmsis_nn
  PUBLIC micro_base
  PUBLIC micro_framework
  PUBLIC micro_ops
  PRIVATE CMSISNN
)
