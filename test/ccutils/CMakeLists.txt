include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB MACE_CC_TEST_UTILS_SRCS
  mace/ops/*.cc
)

add_library(mace_cc_test_utils ${MACE_CC_TEST_UTILS_SRCS})
target_link_libraries(mace_cc_test_utils PUBLIC
  ops
)
