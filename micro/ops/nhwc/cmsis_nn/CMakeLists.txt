add_library(micro_ops_nhwc_cmsis_nn
  arm_conv_2d_int8.cc
  arm_pooling_int8.cc
  arm_softmax_int8.cc
  arm_mat_mul_int8.cc
  arm_eltwise_int8.cc
  arm_depthwise_conv_2d_int8.cc
  dequantize.cc
  quantize.cc
  utilities.cc
)

target_link_libraries(micro_ops_nhwc_cmsis_nn
  PRIVATE micro_base
  PRIVATE CMSISNN
)
