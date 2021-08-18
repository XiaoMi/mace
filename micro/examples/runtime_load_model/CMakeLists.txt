
if(NOT MICRO_MODEL_NAME OR NOT MICRO_DATA_NAME)
  message(FATAL_ERROR "MICRO_MODEL_NAME or MICRO_DATA_NAME is undefined")
endif()

add_executable(runtime_load_${MICRO_MODEL_NAME} main.cc)
target_link_libraries(runtime_load_${MICRO_MODEL_NAME} model_${MICRO_MODEL_NAME} micro)
target_include_directories(runtime_load_${MICRO_MODEL_NAME} PRIVATE ../../codegen ..)
target_compile_options(${MICRO_MODEL_NAME}
  PRIVATE "-Wno-error"
  PRIVATE "-Wno-unused-variable"
)
target_compile_definitions(runtime_load_${MICRO_MODEL_NAME} PRIVATE "-DMICRO_MODEL_NAME=${MICRO_MODEL_NAME}")
target_compile_definitions(runtime_load_${MICRO_MODEL_NAME} PRIVATE "-DMICRO_DATA_NAME=${MICRO_DATA_NAME}")

