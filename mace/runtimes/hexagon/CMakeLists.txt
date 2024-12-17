set(HEXAGON_SRCS
  hexagon_base_runtime.cc
)

if(MACE_ENABLE_HEXAGON_DSP)
  set(HEXAGON_SRCS
    ${HEXAGON_SRCS}
    dsp/hexagon_dsp_runtime.cc
    dsp/hexagon_dsp_wrapper.cc
  )
endif(MACE_ENABLE_HEXAGON_DSP)

if(MACE_ENABLE_HEXAGON_HTA)
  set(HEXAGON_SRCS
    ${HEXAGON_SRCS}
    hta/hexagon_hta_runtime.cc
    hta/hexagon_hta_transformer.cc
    hta/hexagon_hta_wrapper.cc
  )
  if(MACE_ENABLE_OPENCL)
      set(HEXAGON_SRCS ${HEXAGON_SRCS} hta/opencl/hexagon_hta_opencl_runtime.cc)
  endif(MACE_ENABLE_OPENCL)
endif(MACE_ENABLE_HEXAGON_HTA)


add_library(hexagon_runtime STATIC ${HEXAGON_SRCS})
if(MACE_ENABLE_OPENCL)
  add_dependencies(hexagon_runtime opencl_clhpp)
endif(MACE_ENABLE_OPENCL)
target_link_libraries(hexagon_runtime core hexagon_controller)
install(TARGETS hexagon_runtime ARCHIVE DESTINATION lib)
