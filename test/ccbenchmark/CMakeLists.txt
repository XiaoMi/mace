include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

if(QNX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lgomp -Wno-error=unknown-pragmas")
else()
  if (NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
  endif()
endif()

file(GLOB MACE_BENCHMARK_TEST_SRCS
  mace/benchmark_utils/*.cc
  mace/ops/*.cc
)
add_executable(mace_cc_benchmark ${MACE_BENCHMARK_TEST_SRCS})
target_link_libraries(mace_cc_benchmark PUBLIC
  mace_cc_test_utils
  mace_static
  gflags
)
add_dependencies(mace_cc_benchmark eigen3)

install(TARGETS mace_cc_benchmark RUNTIME DESTINATION bin)
