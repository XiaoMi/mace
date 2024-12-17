set(FLOWS_SRCS flow_registry.cc)

add_subdirectory(cpu)

if(MACE_ENABLE_OPENCL)
  add_subdirectory(opencl)
endif(MACE_ENABLE_OPENCL)

if(MACE_ENABLE_HEXAGON)
  add_subdirectory(hexagon)
endif(MACE_ENABLE_HEXAGON)

if(MACE_ENABLE_MTK_APU)
  add_subdirectory(apu)
endif(MACE_ENABLE_MTK_APU)

if(MACE_ENABLE_QNN)
  add_subdirectory(qnn)
endif(MACE_ENABLE_QNN)

add_library(flows STATIC ${FLOWS_SRCS})
target_link_libraries(flows PRIVATE core cpu_flow)
if(MACE_ENABLE_OPENCL)
  target_link_libraries(flows PRIVATE opencl_flow)
endif(MACE_ENABLE_OPENCL)
if(MACE_ENABLE_HEXAGON)
  target_link_libraries(flows PRIVATE hexagon_flow)
endif(MACE_ENABLE_HEXAGON)
install(TARGETS flows ARCHIVE DESTINATION lib)
