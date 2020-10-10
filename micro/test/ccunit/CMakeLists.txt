add_executable(micro_ops_test
  micro/ops/stack_test.cc
  micro/ops/reshape_test.cc
  micro/ops/eltwise_test.cc
  micro/ops/nhwc/pooling_test.cc
  micro/ops/nhwc/conv_2d_test.cc
  micro/ops/nhwc/depthwise_conv_2d_opt_test.cc
  micro/ops/nhwc/batch_norm_test.cc
  micro/ops/nhwc/depthwise_conv_2d_test.cc
  micro/ops/nhwc/conv_2d_opt_test.cc
  micro/ops/squeeze_test.cc
  micro/ops/reduce_test.cc
  micro/ops/strided_slice_test.cc
  micro/ops/matmul_test.cc
  micro/ops/shape_test.cc
  micro/ops/activation_test.cc
  micro/ops/argmax_test.cc
  micro/ops/softmax_test.cc
  micro/ops/bias_add_test.cc
  micro/ops/expand_dims_test.cc
  micro/ops/concat_test.cc
)
target_link_libraries(micro_ops_test
  PRIVATE micro_base
  PRIVATE micro_ops_for_test
  PRIVATE micro_ccutils
  PRIVATE gtest
  PRIVATE gtest_main
)

if(MICRO_MODEL_NAME)
  add_executable(micro_cc_test
    micro/model/net_def_test.cc
    micro/framework/graph_test.cc
    micro/codegen/engine_test.cc
  )
  target_link_libraries(micro_cc_test
    micro
    models
    gtest
    gtest_main
  )
  target_compile_definitions(micro_cc_test PRIVATE "-DMICRO_MODEL_NAME=${MICRO_MODEL_NAME}")
endif()
