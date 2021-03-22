set(CORE_SRCS
  kv_storage.cc
  net_def_adapter.cc
  net_optimizer.cc
  quantize.cc
  runtime_failure_mock.cc
  tensor.cc
  types.cc
  workspace.cc
  flow/base_flow.cc
  flow/common_fp32_flow.cc
  flow/flow_registry.cc
  memory/general_memory_manager.cc
  memory/rpcmem/rpcmem.cc
  net/allocate_opt_strategy.cc
  net/allocate_ref_strategy.cc
  net/serial_net.cc
  ops/op_construct_context.cc
  ops/op_condition_builder.cc
  ops/ops_utils.cc
  ops/operator.cc
  ops/op_init_context.cc
  ops/op_condition_context.cc
  ops/op_context.cc
  proto/net_def_helper.cc
  proto/arg_helper.cc
  registry/ops_registry.cc
  registry/op_registration_info.cc
  registry/op_delegator_registry.cc
  runtime/runtime.cc
  runtime/runtime_registry.cc
)

add_library(extra_link_libs_target INTERFACE)

if(MACE_ENABLE_RPCMEM)
  target_link_libraries(extra_link_libs_target INTERFACE rpcmems)
endif(MACE_ENABLE_RPCMEM)

add_library(core STATIC ${CORE_SRCS})
target_link_libraries(core PRIVATE
  proto
  utils
  port
  generated_version
  extra_link_libs_target
)

add_dependencies(core half)

if(MACE_ENABLE_QUANTIZE)
  add_dependencies(core gemmlowp)
endif(MACE_ENABLE_QUANTIZE)
