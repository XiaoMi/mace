if(HEXAGON OR HEXAGON_STUB)
  execute_process(
    COMMAND mkdir -p ${CMAKE_CURRENT_SOURCE_DIR}/codegen
    COMMAND ${HEXAGON_SDK_ROOT}/tools/qaic/Ubuntu16/qaic
    -mdll -o ${CMAKE_CURRENT_SOURCE_DIR}/codegen
    -I${HEXAGON_SDK_ROOT}/libs/fastcv/dspCV/android_Release/ship
    -I${HEXAGON_SDK_ROOT}/libs/common/rpcmem/android_Release/ship
    -I${HEXAGON_SDK_ROOT}/libs/common/adspmsgd/ship/android_Release
    -I${HEXAGON_SDK_ROOT}/incs
    -I${HEXAGON_SDK_ROOT}/libs/common/remote/ship/android_Release
    -I${HEXAGON_SDK_ROOT}/incs/stddef
    ${CMAKE_CURRENT_SOURCE_DIR}/micro/rpc/benchmark.idl
    WORKING_DIRECTORY .
  )
endif()

add_library(micro_benchmark_utils
  micro/benchmark_utils/test_benchmark.cc
)
target_include_directories(micro_benchmark_utils PUBLIC .)
target_link_libraries(micro_benchmark_utils
  PUBLIC micro_base
  PUBLIC micro_ccutils
)

add_library(micro_benchmark_lib
  micro/ops/reduce_benchmark.cc
  micro/ops/eltwise_benchmark.cc
  micro/ops/activation_benchmark.cc
  micro/ops/matmul_benchmark.cc
  micro/ops/softmax_benchmark.cc
  micro/ops/bias_add_benchmark.cc
  micro/ops/nhwc/conv_2d_benchmark.cc
  micro/ops/nhwc/pooling_benchmark.cc
  micro/ops/nhwc/conv_2d_opt_benchmark.cc
  micro/ops/nhwc/batch_norm_benchmark.cc
  micro/ops/nhwc/depthwise_conv_2d_benchmark.cc
  micro/ops/nhwc/depthwise_conv_2d_opt_benchmark.cc
)
target_link_libraries(micro_benchmark_lib
  PUBLIC micro_base
  PUBLIC micro_ccutils
  PUBLIC micro_ops_for_test
  PUBLIC micro_framework_for_optest
  PUBLIC micro_benchmark_utils
)
target_include_directories(micro_benchmark_lib PUBLIC .)

if(NOT HEXAGON_STUB AND NOT HEXAGON)
  add_executable(micro_cc_benchmark
    micro/test_benchmark_main.cc
  )
  target_link_libraries(micro_cc_benchmark
    PRIVATE micro_base
    PRIVATE micro_ccutils
    PRIVATE micro_ops_for_test
    PRIVATE micro_framework_for_optest
    PRIVATE micro_benchmark_utils
    -Wl,--whole-archive
    PUBLIC micro_benchmark_lib
    -Wl,--no-whole-archive
  )
endif()

if(HEXAGON_STUB)
  add_library(benchmark_idl_stub
    codegen/benchmark_stub.c
  )
  target_link_libraries(benchmark_idl_stub
    PUBLIC hexagon_sdk_headers_arm
    PUBLIC hexagon_sdk_cdsprpc
  )

  add_library(benchmark_stub
    micro/rpc/stub/benchmark.cc
  )
  target_link_libraries(benchmark_stub
    PUBLIC benchmark_idl_stub
    PUBLIC micro_rpc_stub
  )
  target_include_directories(benchmark_stub
    PUBLIC .
    PUBLIC ${PROJECT_SOURCE_DIR}
  )

  add_executable(micro_cc_benchmark
    micro/test_benchmark_main.cc
  )
  target_link_libraries(micro_cc_benchmark
    benchmark_stub
  )
  if(ANDROID)
    target_link_libraries(micro_cc_benchmark log)
  endif()
endif()

if(HEXAGON)
  add_library(benchmark_idk_skel
    codegen/benchmark_skel.c
  )
  target_link_libraries(benchmark_idk_skel hexagon_sdk_headers_dsp)
  target_compile_options(benchmark_idk_skel PRIVATE -fPIC)

  add_library(benchmark_skel SHARED
    micro/rpc/skel/benchmark.c
  )
  target_link_libraries(benchmark_skel
    benchmark_idk_skel
    micro_benchmark_utils
    -Wl,--whole-archive
    micro_benchmark_lib
    -Wl,--no-whole-archive
  )
endif()
