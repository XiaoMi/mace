add_executable(micro_run_static micro_run.cc)
target_link_libraries(micro_run_static micro_all_ops micro_base models  gflags)
target_compile_options(micro_run_static PRIVATE "-std=c++11")
target_compile_definitions(micro_run_static PRIVATE "-DMICRO_MODEL_NAME=${MICRO_MODEL_NAME}")
