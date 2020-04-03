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
  ops/op_condition_builder.cc
  ops/op_condition_context.cc
  ops/op_construct_context.cc
  ops/op_context.cc
  ops/operator.cc
  ops/op_init_context.cc
  quantize.cc
  registry/op_delegator_registry.cc
  registry/op_registration_info.cc
  registry/ops_registry.cc
  runtime_failure_mock.cc
  types.cc
  workspace.cc
  runtime/cpu/cpu_runtime.cc
)

if(MACE_ENABLE_RPCMEM)
  set(CORE_SRCS ${CORE_SRCS} rpcmem.cc)
  set(EXTRA_LINK_LIBS ${EXTRA_LINK_LIBS} rpcmem)
endif(MACE_ENABLE_RPCMEM)

if(MACE_ENABLE_OPENCL)
  set(CORE_SRCS ${CORE_SRCS}
    runtime/opencl/gpu_device.cc
    runtime/opencl/gpu_runtime.cc
    runtime/opencl/opencl_allocator.cc
    runtime/opencl/opencl_helper.cc
    runtime/opencl/opencl_runtime.cc
    runtime/opencl/opencl_util.cc
    runtime/opencl/opencl_wrapper.cc
    runtime/opencl/scratch_image.cc
  )
  set(EXTRA_LINK_LIBS ${EXTRA_LINK_LIBS} generated_opencl_kernel dl)
endif(MACE_ENABLE_OPENCL)

if(MACE_ENABLE_HEXAGON_DSP OR MACE_ENABLE_HEXAGON_HTA)
  set(CORE_SRCS ${CORE_SRCS}
    runtime/hexagon/hexagon_allocator.cc
    runtime/hexagon/hexagon_device.cc
  )
endif(MACE_ENABLE_HEXAGON_DSP OR MACE_ENABLE_HEXAGON_HTA)

if(MACE_ENABLE_HEXAGON_DSP)
  set(CORE_SRCS ${CORE_SRCS} runtime/hexagon/hexagon_dsp_wrapper.cc)
  set(EXTRA_LINK_LIBS ${EXTRA_LINK_LIBS} hexagon_controller)
endif(MACE_ENABLE_HEXAGON_DSP)

if(MACE_ENABLE_HEXAGON_HTA)
  set(CORE_SRCS ${CORE_SRCS} runtime/hexagon/hexagon_hta_transformer.cc)
  set(CORE_SRCS ${CORE_SRCS} runtime/hexagon/hexagon_hta_wrapper.cc)
  set(EXTRA_LINK_LIBS ${EXTRA_LINK_LIBS} hta_hexagon_runtime)
endif(MACE_ENABLE_HEXAGON_HTA)

if(MACE_ENABLE_MTK_APU)
  set(CORE_SRCS ${CORE_SRCS} runtime/apu/apu_wrapper.cc)
  set(EXTRA_LINK_LIBS ${EXTRA_LINK_LIBS} apu-frontend)
endif(MACE_ENABLE_MTK_APU)

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
