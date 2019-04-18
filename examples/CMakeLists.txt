file(GLOB MACE_EXAMPLE_SRCS
  cli/example.cc
)
add_executable(mace_example ${MACE_EXAMPLE_SRCS})
target_link_libraries(mace_example PUBLIC
  mace_static
  gflags
)

install(TARGETS mace_example RUNTIME DESTINATION bin)
