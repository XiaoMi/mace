file(GLOB MACE_RUN_SRCS
  mace_run.cc
)
add_executable(mace_run ${MACE_RUN_SRCS})
target_link_libraries(mace_run PUBLIC
  mace_static
  model
  gflags
)

install(TARGETS mace_run RUNTIME DESTINATION bin)
