if(NOT MICRO_MODEL_NAME)
message(FATAL_ERROR "MICRO_MODEL_NAME is undefined")
endif()

add_executable(micro_run_static micro_run.cc)
target_link_libraries(micro_run_static micro_port model_${MICRO_MODEL_NAME} micro gflags)
target_compile_options(micro_run_static PRIVATE "-std=c++11")
target_compile_definitions(micro_run_static PRIVATE "-DMICRO_MODEL_NAME=${MICRO_MODEL_NAME}")
