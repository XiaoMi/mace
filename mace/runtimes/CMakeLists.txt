set(RUNTIMES_SRCS STATIC runtime_registry.cc)

add_subdirectory(cpu)

if(MACE_ENABLE_OPENCL)
  add_subdirectory(opencl)
endif(MACE_ENABLE_OPENCL)

if(MACE_ENABLE_HEXAGON)
  add_subdirectory(hexagon)
endif(MACE_ENABLE_HEXAGON)

if(DMACE_ENABLE_MTK_APU)
  add_subdirectory(apu)
endif(DMACE_ENABLE_MTK_APU)

if(DMACE_ENABLE_QNN)
  add_subdirectory(qnn)
endif(DMACE_ENABLE_QNN)

add_library(runtimes STATIC ${RUNTIMES_SRCS})
target_link_libraries(runtimes PRIVATE core cpu_runtime)
if(MACE_ENABLE_OPENCL)
  target_link_libraries(runtimes PRIVATE opencl_runtime)
endif(MACE_ENABLE_OPENCL)
if(MACE_ENABLE_HEXAGON)
  target_link_libraries(runtimes PRIVATE hexagon_runtime)
endif(MACE_ENABLE_HEXAGON)
install(TARGETS runtimes ARCHIVE DESTINATION lib)
