// Copyright 2020 The MACE Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mace/libmace/engines/single_flow_engine.h"

#include <utility>

#include "mace/core/flow/flow_registry.h"
#include "mace/core/runtime/runtime_registry.h"
#include "mace/proto/mace.pb.h"

namespace mace {
SingleFlowEngine::SingleFlowEngine(const MaceEngineConfig &config)
    : BaseEngine(config) {
  LOG(INFO) << "Creating SingleFlowEngine, MACE version: " << MaceVersion();
}

SingleFlowEngine::~SingleFlowEngine() {}

MaceStatus SingleFlowEngine::InitRuntime(const NetDef *net_def) {
  auto target_runtime_type =
      static_cast<RuntimeType>(ProtoArgHelper::GetOptionalArg<NetDef, int>(
          *net_def, "runtime_type", static_cast<int>(RuntimeType::RT_NONE)));
  MACE_CHECK(target_runtime_type != RuntimeType::RT_NONE,
             "no runtime type specified");

  // create runtime
  auto runtime_registry = make_unique<RuntimeRegistry>();
  RegisterAllRuntimes(runtime_registry.get());

  auto runtime = SmartCreateRuntime(
      runtime_registry.get(), target_runtime_type, runtime_context_.get());
  runtime_ = std::move(runtime);
  if (runtime_->GetRuntimeType() != RT_CPU) {
    auto cpu_runtime = SmartCreateRuntime(
        runtime_registry.get(), RuntimeType::RT_CPU, runtime_context_.get());
    cpu_runtime_ = std::move(cpu_runtime);
    VLOG(3) << "create two runtimes, main: " << runtime_->GetRuntimeType();
  } else {
    cpu_runtime_ = runtime_;
    VLOG(3) << "only CPU runtime.";
  }

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus SingleFlowEngine::Init(
    const NetDef *net_def, const std::vector<std::string> &input_nodes,
    const std::vector<std::string> &output_nodes,
    const unsigned char *model_data,
    const int64_t model_data_size, bool *model_data_unused) {
  VLOG(1) << "Initializing SingleFlowEngine";
  MACE_RETURN_IF_ERROR(InitRuntime(net_def));
  MACE_RETURN_IF_ERROR(BaseEngine::Init(
      net_def, input_nodes, output_nodes,
      model_data, model_data_size, model_data_unused));
  // init runtimes
  auto mem_type_i = ProtoArgHelper::GetOptionalArg<NetDef, int>(
      *net_def, "opencl_mem_type",
      static_cast<MemoryType>(MemoryType::GPU_IMAGE));
  auto mem_type = static_cast<MemoryType>(mem_type_i);
  MACE_RETURN_IF_ERROR(runtime_->Init(config_impl_, mem_type));
  if (cpu_runtime_ != runtime_) {
    MACE_RETURN_IF_ERROR(cpu_runtime_->Init(config_impl_, mem_type));
  }

  // create FlowRegistry
  auto flow_registry = make_unique<FlowRegistry>();
  RegisterAllFlows(flow_registry.get());

  // create and init flow
  auto flow_context = make_unique<FlowContext>(
      config_impl_, op_registry_.get(), op_delegator_registry_.get(),
      cpu_runtime_.get(), runtime_.get(), thread_pool_.get(), this);
  DataType data_type = static_cast<DataType>(net_def->data_type());
  FlowSubType sub_type = (data_type == DataType::DT_BFLOAT16) ?
                         FlowSubType::FW_SUB_BF16 : FlowSubType::FW_SUB_REF;
  RuntimeType runtime_type = runtime_->GetRuntimeType();
  single_flow_ = flow_registry->CreateFlow(runtime_type, sub_type,
                                           flow_context.get());
  MACE_RETURN_IF_ERROR(single_flow_->Init(net_def, model_data, model_data_size,
                                          model_data_unused));
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus SingleFlowEngine::Init(const MultiNetDef *multi_net_def,
                                  const std::vector<std::string> &input_nodes,
                                  const std::vector<std::string> &output_nodes,
                                  const unsigned char *model_data,
                                  const int64_t model_data_size,
                                  bool *model_data_unused) {
  const NetDef &net_def = multi_net_def->net_def(0);
  return Init(&net_def, input_nodes, output_nodes,
              model_data, model_data_size, model_data_unused);
}

MaceStatus SingleFlowEngine::Run(
    const std::map<std::string, MaceTensor> &inputs,
    std::map<std::string, MaceTensor> *outputs,
    RunMetadata *run_metadata) {
  return single_flow_->Run(inputs, outputs, run_metadata);
}

}  // namespace mace
