file(GLOB MACE_RUN_SRCS
  mace_run.cc
)
add_executable(mace_run ${MACE_RUN_SRCS})
target_link_libraries(mace_run
  mace_static
  model
  extra_link_libs_target
  gflags
)
if(NOT ANDROID)
  target_link_libraries(mace_run pthread)
endif()

if(MACE_ENABLE_HEXAGON_DSP)
  target_link_libraries(mace_run hexagon_controller)
endif()

install(TARGETS mace_run RUNTIME DESTINATION bin)
