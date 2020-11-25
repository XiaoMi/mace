
if(NOT MICRO_MODEL_NAME OR NOT MICRO_DATA_NAME)
  message(FATAL_ERROR "MICRO_MODEL_NAME or MICRO_DATA_NAME is undefined")
endif()

add_executable(${MICRO_MODEL_NAME}-test main.cc)
target_compile_options(${MICRO_MODEL_NAME}-test PRIVATE "-Wno-error")
target_link_libraries(${MICRO_MODEL_NAME}-test micro ${MICRO_MODEL_NAME})
target_compile_definitions(${MICRO_MODEL_NAME}-test PRIVATE "-DMICRO_MODEL_NAME=${MICRO_MODEL_NAME}")
target_compile_definitions(${MICRO_MODEL_NAME}-test PRIVATE "-DMICRO_DATA_NAME=${MICRO_DATA_NAME}")
