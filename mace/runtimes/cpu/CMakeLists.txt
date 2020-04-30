set(CPU_SRCS
  cpu_ref_allocator.cc
  cpu_ref_runtime.cc
  cpu_runtime.cc
)

if(MACE_ENABLE_RPCMEM)
  set(CPU_SRCS
    ${CPU_SRCS}
    ion/cpu_ion_allocator.cc
    ion/cpu_ion_runtime.cc
  )
endif(MACE_ENABLE_RPCMEM)

add_library(cpu_runtime STATIC ${CPU_SRCS})
target_link_libraries(cpu_runtime core)
install(TARGETS cpu_runtime ARCHIVE DESTINATION lib)