if(HEXAGON OR HEXAGON_STUB)
  execute_process(
    COMMAND mkdir -p ${CMAKE_CURRENT_SOURCE_DIR}/codegen
    COMMAND ${HEXAGON_SDK_ROOT}/tools/qaic/Ubuntu16/qaic
    -mdll -o ${CMAKE_CURRENT_SOURCE_DIR}/codegen
    -I${HEXAGON_SDK_ROOT}/libs/fastcv/dspCV/android_Debug/ship
    -I${HEXAGON_SDK_ROOT}/libs/common/rpcmem/android_Debug/ship
    -I${HEXAGON_SDK_ROOT}/libs/common/adspmsgd/ship/android_Debug
    -I${HEXAGON_SDK_ROOT}/incs
    -I${HEXAGON_SDK_ROOT}/libs/common/remote/ship/android_Debug
    -I${HEXAGON_SDK_ROOT}/incs/stddef
    ${CMAKE_CURRENT_SOURCE_DIR}/macemc/rpc/macemc.idl
    WORKING_DIRECTORY .
  )
endif()

if(HEXAGON AND MICRO_MODEL_NAME)
  add_library(macemc_idl_skel
    codegen/macemc_skel.c
  )
  target_link_libraries(macemc_idl_skel  hexagon_sdk_headers_dsp)
  target_compile_options(macemc_idl_skel PRIVATE "-fPIC")

  add_library(macemc_skel SHARED
    macemc/rpc/skel/macemc.cc
    macemc/rpc/skel/micro_tester.cc
  )
  target_compile_definitions(macemc_skel PUBLIC "-DMICRO_MODEL_NAME=${MICRO_MODEL_NAME}")
  target_link_libraries(macemc_skel
    micro_engine
    micro_rpc_skel
    hexagon_sdk_headers_dsp
    -Wl,--whole-archive
    macemc_idl_skel
    -Wl,--no-whole-archive
  )
endif()

if(HEXAGON_STUB)
  add_library(macemc_idl_stub
  codegen/macemc_stub.c
  )
  target_link_libraries(macemc_idl_stub hexagon_sdk_headers_arm hexagon_sdk_cdsprpc)

  add_library(macemc_stub
  macemc/rpc/stub/macemc.cc
  )
  target_include_directories(macemc_stub PUBLIC . )
  target_link_libraries(macemc_stub)
  target_link_libraries(macemc_stub macemc_idl_stub micro_rpc_stub)

  add_executable(micro_cc_baseline
  test_baseline_main.cc
  )
  target_link_libraries(micro_cc_baseline macemc_stub)
  if(ANDROID)
  target_link_libraries(micro_cc_baseline log)
  endif()
endif()
