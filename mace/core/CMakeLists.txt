set(CORE_SRCS
  allocator.cc
  arg_helper.cc
  device.cc
  device_context.cc
  kv_storage.cc
  memory_optimizer.cc
  net.cc
  net_def_adapter.cc
  net_optimizer.cc
  op_context.cc
  operator.cc
  quantize.cc
  runtime_failure_mock.cc
  types.cc
  workspace.cc
  runtime/cpu/cpu_runtime.cc
)

if(MACE_ENABLE_OPENCL)
  set(CORE_SRCS ${CORE_SRCS}
    runtime/opencl/gpu_device.cc
    runtime/opencl/gpu_runtime.cc
    runtime/opencl/opencl_allocator.cc
    runtime/opencl/opencl_runtime.cc
    runtime/opencl/opencl_util.cc
    runtime/opencl/opencl_wrapper.cc
    runtime/opencl/scratch_image.cc
  )
  set(EXTRA_LINK_LIBS ${EXTRA_LINK_LIBS} generated_opencl_kernel dl)
endif(MACE_ENABLE_OPENCL)

if(MACE_ENABLE_HEXAGON_DSP)
  set(CORE_SRCS ${CORE_SRCS} runtime/hexagon/hexagon_dsp_wrapper.cc)
  set(EXTRA_LINK_LIBS ${EXTRA_LINK_LIBS} hexagon_controller)
endif(MACE_ENABLE_HEXAGON_DSP)

if(MACE_ENABLE_HEXAGON_HTA)
  set(CORE_SRCS ${CORE_SRCS} runtime/hexagon/hexagon_hta_wrapper.cc)
  set(EXTRA_LINK_LIBS ${EXTRA_LINK_LIBS} hta_controller hta_hexagon_runtime npu)
endif(MACE_ENABLE_HEXAGON_HTA)

add_library(core STATIC ${CORE_SRCS})
target_link_libraries(core PRIVATE
  proto
  utils
  port
  generated_version
  ${EXTRA_LINK_LIBS}
)

add_dependencies(core half)

if(MACE_ENABLE_OPENCL)
  add_dependencies(core opencl_clhpp)
endif(MACE_ENABLE_OPENCL)

if(MACE_ENABLE_QUANTIZE)
  add_dependencies(core gemmlowp)
endif(MACE_ENABLE_QUANTIZE)

install(TARGETS core ARCHIVE DESTINATION lib)
