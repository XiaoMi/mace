set(APU_SRCS apu_runtime.cc)

if(MACE_MTK_APU_VERSION EQUAL 1)
  set(APU_SRCS ${APU_SRCS} runtime/apu/v1v2v3/v1/apu_wrapper.cc)
  set(EXTRA_LINK_LIBS ${EXTRA_LINK_LIBS} apu-frontend)
elseif(MACE_MTK_APU_VERSION EQUAL 2)
  set(APU_SRCS ${APU_SRCS} runtime/apu/v1v2v3/v2v3/apu_wrapper.cc)
  set(EXTRA_LINK_LIBS ${EXTRA_LINK_LIBS} apu-frontend)
elseif(MACE_MTK_APU_VERSION EQUAL 3)
  set(APU_SRCS ${APU_SRCS} runtime/apu/v1v2v3/v2v3/apu_wrapper.cc)
  set(EXTRA_LINK_LIBS ${EXTRA_LINK_LIBS} apu-frontend)
elseif(MACE_MTK_APU_VERSION EQUAL 4)
  set(APU_SRCS ${APU_SRCS}
    runtime/apu/v4/apu_wrapper.cc
    runtime/apu/v4/neuron_delegate_kernel.cc
    runtime/apu/v4/neuron_implementation.cc
  )
else()
  message(FATAL_ERROR "Invalid MTK APU version.")
endif()

add_library(apu_runtime STATIC ${APU_SRCS})

set(EXTRA_LINK_LIBS apu-frontend)
target_link_libraries(apu_runtime core ${EXTRA_LINK_LIBS})

install(TARGETS apu_runtime ARCHIVE DESTINATION lib)