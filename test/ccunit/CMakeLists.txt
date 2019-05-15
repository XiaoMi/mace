include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB MACE_CC_TEST_SRCS
  mace/utils/*.cc
  mace/port/*.cc
  mace/ops/*.cc
  mace/libmace/*.cc
)

add_executable(mace_cc_test ${MACE_CC_TEST_SRCS})
target_link_libraries(mace_cc_test PUBLIC
  mace_cc_test_utils
  mace_static
  gtest
  gtest_main
)

install(TARGETS mace_cc_test RUNTIME DESTINATION bin)
