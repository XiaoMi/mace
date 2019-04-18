file(GLOB MACE_RUN_SRCS
  validation/mace_run.cc
)
add_executable(mace_run ${MACE_RUN_SRCS})
target_link_libraries(mace_run PUBLIC
  mace_static
  gflags
)


file(GLOB MACE_BENCHMARK_MODEL_SRCS
  benchmark/benchmark_model.cc
)
add_executable(benchmark_model ${MACE_BENCHMARK_MODEL_SRCS})
target_link_libraries(benchmark_model PUBLIC
  mace_static
  gflags
)

file(GLOB MACE_BENCHMARK_MODEL_THROUGHPUT_SRCS
  benchmark/benchmark_model_throughput.cc
)
add_executable(benchmark_model_throughput ${MACE_BENCHMARK_MODEL_THROUGHPUT_SRCS})
target_link_libraries(benchmark_model_throughput PUBLIC
  mace_static
  gflags
)

install(TARGETS mace_run RUNTIME DESTINATION bin)
install(TARGETS benchmark_model RUNTIME DESTINATION bin)
install(TARGETS benchmark_model_throughput RUNTIME DESTINATION bin)
