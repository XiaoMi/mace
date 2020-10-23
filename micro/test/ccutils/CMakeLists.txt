add_library(micro_ccutils
  micro/ops/test_utils.cc
  micro/ops/operator.test.cc
  micro/ops/substitute_op.cc
  micro/common/global_buffer.cc
)

target_include_directories(micro_ccutils PUBLIC .)
target_link_libraries(micro_ccutils micro_base micro_framework_for_optest)
target_compile_options(micro_ccutils PUBLIC "-std=c++11")

if(HEXAGON_STUB)
  add_library(micro_rpc_stub
    rpc/stub/base_handle.cc
  )
  target_include_directories(micro_rpc_stub PUBLIC .)
  target_link_libraries(micro_rpc_stub PUBLIC hexagon_sdk_headers_arm)
endif()

if(HEXAGON)
  add_library(micro_rpc_skel
    rpc/skel/base_func.cc
  )
  target_include_directories(micro_rpc_skel PUBLIC .)
  target_link_libraries(micro_rpc_skel
    PUBLIC hexagon_sdk_headers_dsp
  )
  target_compile_options(micro_rpc_skel PUBLIC "-fPIC")
endif()
