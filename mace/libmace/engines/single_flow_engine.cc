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

MaceStatus SingleFlowEngine::CreateAndInitRuntimes(const NetDef *net_def,
                                                   BaseEngine *tutor) {
  auto runtime_registry = make_unique<RuntimeRegistry>();
  RegisterAllRuntimes(runtime_registry.get());

  auto target_runtime_type =
      static_cast<RuntimeType>(ProtoArgHelper::GetOptionalArg<NetDef, int>(
          *net_def, "runtime_type", static_cast<int>(RuntimeType::RT_NONE)));
  MACE_CHECK(target_runtime_type != RuntimeType::RT_NONE,
             "no runtime type specified");

  // Get runtimes from tutor if tutor exist
  auto cpu_rt_key = (RuntimeType::RT_CPU << 16) | CPU_BUFFER;
  if (tutor != nullptr) {
    auto &tutor_runtimes = GetRuntimesOfTutor(tutor);
    if (tutor_runtimes.count(cpu_rt_key) != 0) {
      cpu_runtime_ = tutor_runtimes.at(cpu_rt_key);
    }
  }
  if (cpu_runtime_ == nullptr) {
    auto cpu_runtime = SmartCreateRuntime(
        runtime_registry.get(), RuntimeType::RT_CPU, runtime_context_.get());
    MACE_RETURN_IF_ERROR(cpu_runtime->Init(config_impl_.get(), CPU_BUFFER));
    cpu_runtime_ = std::move(cpu_runtime);
  }
  runtimes_.emplace(cpu_rt_key, cpu_runtime_);

  if (target_runtime_type == RT_CPU) {
    runtime_ = cpu_runtime_;
    return MaceStatus::MACE_SUCCESS;
  }

  // Create target runtime
  MACE_CHECK(target_runtime_type == RT_OPENCL, "Only support OpenCL now");
  auto mem_type_i = ProtoArgHelper::GetOptionalArg<NetDef, int>(
      *net_def, "opencl_mem_type",
      static_cast<MemoryType>(MemoryType::GPU_IMAGE));
  auto mem_type = static_cast<MemoryType>(mem_type_i);
  uint32_t target_rt_key = (target_runtime_type << 16) | mem_type_i;
  if (tutor != nullptr) {
    auto &tutor_runtimes = GetRuntimesOfTutor(tutor);
    if (tutor_runtimes.count(target_rt_key) != 0) {
      runtime_ = tutor_runtimes.at(target_rt_key);
    }
  }
  if (runtime_ == nullptr) {
    auto runtime = SmartCreateRuntime(
        runtime_registry.get(), target_runtime_type, runtime_context_.get());
    MACE_RETURN_IF_ERROR(runtime->Init(config_impl_.get(), mem_type));
    runtime_ = std::move(runtime);
  }
  runtimes_.emplace(target_rt_key, runtime_);

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus SingleFlowEngine::DoInit(
    const NetDef *net_def, const std::vector<std::string> &input_nodes,
    const std::vector<std::string> &output_nodes,
    const unsigned char *model_data, const int64_t model_data_size,
    bool *model_data_unused, BaseEngine *tutor) {
  VLOG(1) << "Initializing SingleFlowEngine";
  MACE_RETURN_IF_ERROR(BaseEngine::Init(
      net_def, input_nodes, output_nodes,
      model_data, model_data_size, model_data_unused));
  MACE_RETURN_IF_ERROR(CreateAndInitRuntimes(net_def, tutor));

  // create FlowRegistry
  auto flow_registry = make_unique<FlowRegistry>();
  RegisterAllFlows(flow_registry.get());

  // create and init flow
  auto flow_context = make_unique<FlowContext>(
      config_impl_.get(), op_registry_.get(), op_delegator_registry_.get(),
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

// @Deprecated, will be removed in future version
MaceStatus SingleFlowEngine::Init(
    const NetDef *net_def, const std::vector<std::string> &input_nodes,
    const std::vector<std::string> &output_nodes,
    const unsigned char *model_data, const int64_t model_data_size,
    bool *model_data_unused) {
  return DoInit(net_def, input_nodes, output_nodes, model_data, model_data_size,
                model_data_unused, nullptr);
}

MaceStatus SingleFlowEngine::Init(const MultiNetDef *multi_net_def,
                                  const std::vector<std::string> &input_nodes,
                                  const std::vector<std::string> &output_nodes,
                                  const unsigned char *model_data,
                                  const int64_t model_data_size,
                                  bool *model_data_unused, BaseEngine *tutor) {
  const NetDef &net_def = multi_net_def->net_def(0);
  return DoInit(&net_def, input_nodes, output_nodes,
                model_data, model_data_size, model_data_unused, tutor);
}

MaceStatus SingleFlowEngine::Run(
    const std::map<std::string, MaceTensor> &inputs,
    std::map<std::string, MaceTensor> *outputs,
    RunMetadata *run_metadata) {
  return single_flow_->Run(inputs, outputs, run_metadata);
}

}  // namespace mace
