set(CPU_SRCS
  cpu_ref_flow.cc
)

if(MACE_ENABLE_FP16)
  set(CPU_SRCS ${CPU_SRCS} cpu_fp16_flow.cc)
endif(MACE_ENABLE_FP16)

if(MACE_ENABLE_BFLOAT16)
  set(CPU_SRCS ${CPU_SRCS} cpu_bf16_flow.cc)
endif(MACE_ENABLE_BFLOAT16)

add_library(cpu_flow STATIC ${CPU_SRCS})
target_link_libraries(cpu_flow core)
install(TARGETS cpu_flow ARCHIVE DESTINATION lib)
