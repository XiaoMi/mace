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


#include "mace/libmace/engines/serial_engine.h"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mace/utils/dbg.h"
#include "mace/core/runtime/runtime.h"
#include "mace/core/runtime/runtime_registry.h"

namespace mace {
SerialEngine::SerialEngine(const MaceEngineConfig &config)
    : BaseEngine(config), inter_mem_released_(false) {
  LOG(INFO) << "Creating SerialEngine, MACE version: " << MaceVersion();
}

SerialEngine::~SerialEngine() {}

// @Deprecated, will be removed in future version
MaceStatus SerialEngine::Init(
    const NetDef *net_def, const std::vector<std::string> &input_nodes,
    const std::vector<std::string> &output_nodes,
    const unsigned char *model_data,
    const int64_t model_data_size, bool *model_data_unused) {
  auto ret = BaseEngine::Init(net_def, input_nodes, output_nodes, model_data,
                              model_data_size, model_data_unused);
  MACE_RETURN_IF_ERROR(ret);

  auto multi_net_def = MultiNetDef();
  mace::NetDef *tmp_net_def = multi_net_def.add_net_def();
  *tmp_net_def = *net_def;
  ret = DoInit(&multi_net_def, input_nodes, output_nodes, model_data,
               model_data_size, model_data_unused, nullptr);
  MACE_RETURN_IF_ERROR(ret);

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus SerialEngine::Init(const MultiNetDef *multi_net_def,
                              const std::vector<std::string> &input_nodes,
                              const std::vector<std::string> &output_nodes,
                              const unsigned char *model_data,
                              const int64_t model_data_size,
                              bool *model_data_unused,
                              BaseEngine* tutor) {
  auto ret = BaseEngine::Init(multi_net_def, input_nodes, output_nodes,
                              model_data, model_data_size, model_data_unused,
                              tutor);
  MACE_RETURN_IF_ERROR(ret);

  ret = DoInit(multi_net_def, input_nodes, output_nodes, model_data,
               model_data_size, model_data_unused, tutor);
  MACE_RETURN_IF_ERROR(ret);

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus SerialEngine::BeforeRun() {
  if (inter_mem_released_) {
    MACE_RETURN_IF_ERROR(AllocateIntermediateBuffer());
    inter_mem_released_ = false;
  }
  return BaseEngine::BeforeRun();
}

MaceStatus SerialEngine::Run(
    const std::map<std::string, MaceTensor> &inputs,
    std::map<std::string, MaceTensor> *outputs,
    RunMetadata *run_metadata) {
  // replace the input and output tensors
  for (auto iter = inputs.begin(); iter != inputs.end(); ++iter) {
    (*(run_helper_[iter->first]))[iter->first] = iter->second;
  }
  for (auto iter = outputs->begin(); iter != outputs->end(); ++iter) {
    (*(run_helper_[iter->first]))[iter->first] = iter->second;
  }

  auto flow_num = flows_.size();
  for (size_t i = 0; i < flow_num; ++i) {
    auto *flow = flows_[i].get();
    VLOG(1) << "start run flow: " << flow->GetName();
    auto ret = flow->Run(*(input_tensors_[flow]), output_tensors_[flow].get(),
                         run_metadata);
    MACE_RETURN_IF_ERROR(ret);
  }

  for (auto iter = inputs.begin(); iter != inputs.end(); ++iter) {
    run_helper_[iter->first]->erase(iter->first);
  }
  for (auto iter = outputs->begin(); iter != outputs->end(); ++iter) {
    run_helper_[iter->first]->erase(iter->first);
  }

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus SerialEngine::AfterRun() {
  for (auto iter = runtimes_.begin(); iter != runtimes_.end(); ++iter) {
    iter->second->OnIntermediateBufferUsed(this);
  }
  return BaseEngine::AfterRun();
}

MaceStatus SerialEngine::FakeWarmup() {
  auto flow_num = flows_.size();
  for (size_t i = 0; i < flow_num; ++i) {
    auto *flow = flows_[i].get();
    auto ret = flow->FakeWarmup();
    MACE_RETURN_IF_ERROR(ret);
  }
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus SerialEngine::ReleaseIntermediateBuffer() {
  if (inter_mem_released_) {
    return MaceStatus::MACE_SUCCESS;
  }
  for (auto iter = runtimes_.begin(); iter != runtimes_.end(); ++iter) {
    iter->second->ReleaseIntermediateBuffer(this);
  }
  inter_mem_released_ = true;

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus SerialEngine::AllocateIntermediateBuffer() {
  if (!inter_mem_released_) {
    return MaceStatus::MACE_SUCCESS;
  }
  for (auto &flow : flows_) {
    MACE_RETURN_IF_ERROR(flow->AllocateIntermediateBuffer());
  }
  for (auto iter = runtimes_.begin(); iter != runtimes_.end(); ++iter) {
    iter->second->OnAllocateIntermediateBuffer(this);
  }
  inter_mem_released_ = false;
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus SerialEngine::CreateAndInitRuntimes(
    const NetDefMap &net_defs, NetRuntimeMap *runtime_map, BaseEngine *tutor) {
  // create runtime
  auto runtime_registry = make_unique<RuntimeRegistry>();
  RegisterAllRuntimes(runtime_registry.get());

  // create cpu runtime
  auto cpu_rt_key = (RuntimeType::RT_CPU << 16) | CPU_BUFFER;
  if (tutor != nullptr) {
    auto &tutor_runtimes = GetRuntimesOfTutor(tutor);
    MACE_CHECK(tutor_runtimes.count(cpu_rt_key) > 0,
               "The tutor engine should be initialized first.");
    cpu_runtime_ = tutor_runtimes.at(cpu_rt_key);
  }
  if (cpu_runtime_ == nullptr) {
    auto cpu_runtime = SmartCreateRuntime(
        runtime_registry.get(), RuntimeType::RT_CPU, runtime_context_.get());
    MACE_RETURN_IF_ERROR(cpu_runtime->Init(config_impl_.get(), CPU_BUFFER));
    cpu_runtime_ = std::move(cpu_runtime);
  }
  runtimes_.emplace(cpu_rt_key, cpu_runtime_);

  // Create other runtimes
  for (auto i = net_defs.begin(); i != net_defs.end(); ++i) {
    auto *net_def = i->second;
    auto runtime_type = config_impl_->runtime_type(net_def->name());
    if (runtime_type != RuntimeType::RT_NONE) {
      SetProtoArg(const_cast<NetDef *>(net_def), "runtime_type",
                  static_cast<int>(runtime_type));
    }

    runtime_type =
        static_cast<RuntimeType>(ProtoArgHelper::GetOptionalArg<NetDef, int>(
            *net_def, "runtime_type", static_cast<int>(RuntimeType::RT_NONE)));
    MACE_CHECK(runtime_type != RuntimeType::RT_NONE,
               "no runtime type specified");
    if (runtime_type == RuntimeType::RT_CPU) {
      runtime_map->emplace(net_def, cpu_runtime_);
      continue;
    }

    auto sub_type = SmartGetRuntimeSubType(runtime_type,
                                           runtime_context_.get());
    auto key = (runtime_type << 16) | sub_type;
    std::shared_ptr<Runtime> runtime;
    if (runtimes_.count(key) > 0) {
      runtime = runtimes_.at(key);
    } else {
      if (tutor != nullptr &&
          (runtime_type == RT_CPU || runtime_type == RT_OPENCL)) {
        // TODO(luxuhui): Only support cpu and gpu runtimes now, we can
        //  support hexagon/hta/apu runtimes in the future
        auto &tutor_runtimes = GetRuntimesOfTutor(tutor);
        if (tutor_runtimes.count(key) > 0) {
          runtime = tutor_runtimes.at(key);
        }
      }
      if (runtime == nullptr) {
        auto unique_runtime = SmartCreateRuntime(
            runtime_registry.get(), runtime_type, runtime_context_.get());

        auto mem_type_i =
            static_cast<MemoryType>(ProtoArgHelper::GetOptionalArg<NetDef, int>(
                *net_def, "opencl_mem_type",
                static_cast<int>(MemoryType::MEMORY_NONE)));
        MACE_CHECK(mem_type_i != MemoryType::MEMORY_NONE,
                   "no mem type specified");
        MACE_RETURN_IF_ERROR(unique_runtime->Init(config_impl_.get(),
                                                  mem_type_i));
        runtime = std::move(unique_runtime);
      }
      runtimes_.emplace(key, runtime);
    }
    runtime_map->emplace(net_def, runtime);
  }

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus SerialEngine::CreateAndInitFlows(
    const NetDefMap &net_defs, const NetRuntimeMap &runtime_map,
    const unsigned char *model_data,
    const int64_t model_data_size, bool *model_data_unused) {
  // create FlowRegistry
  auto flow_registry = make_unique<FlowRegistry>();
  RegisterAllFlows(flow_registry.get());

  // create and init flows
  bool flows_data_unused = true;
  for (auto i = net_defs.begin(); i != net_defs.end(); ++i) {
    const NetDef *net_def = i->second;
    auto runtime = runtime_map.at(net_def);
    VLOG(1) << "CreateAndInitFlows, name: " << net_def->name()
            << ", infer_order: " << net_def->infer_order()
            << ", runtime: " << runtime->GetRuntimeType();

    auto flow_context = make_unique<FlowContext>(
        config_impl_.get(), op_registry_.get(), op_delegator_registry_.get(),
        cpu_runtime_.get(), runtime.get(), thread_pool_.get(), this);
    DataType data_type = static_cast<DataType>(net_def->data_type());
    FlowSubType sub_type = (data_type == DataType::DT_BFLOAT16) ?
                           FlowSubType::FW_SUB_BF16 : FlowSubType::FW_SUB_REF;
    RuntimeType runtime_type = runtime->GetRuntimeType();
    auto flow = flow_registry->CreateFlow(runtime_type, sub_type,
                                          flow_context.get());
    bool data_unused = false;
    const auto data_offset = net_def->data_offset();
    auto data_size = net_def->data_size();
    if (data_size == 0) {  // Compatible with old version of NetDef
      data_size = model_data_size;
    }
    MACE_CHECK(data_offset + data_size <= model_data_size);
    MACE_RETURN_IF_ERROR(flow->Init(
        net_def, model_data + data_offset, data_size, &data_unused));
    // In serial engine, we can reuse buffers between flows
    runtime->ReleaseAllBuffer(RENT_SHARE, false);
    if (runtime != cpu_runtime_) {
      cpu_runtime_->ReleaseAllBuffer(RENT_SHARE, false);
    }

    flows_data_unused &= data_unused;
    flows_.push_back(std::move(flow));
  }

  if (model_data_unused != nullptr) {
    *model_data_unused = flows_data_unused;
  }

  return MaceStatus::MACE_SUCCESS;
}

std::unordered_map<std::string, int> SerialEngine::AllocOutTensors(
    const NetDefMap &net_defs, const std::vector<std::string> &glb_out_nodes) {
  // compute the memory needed
  std::multimap<int32_t, int> free_block_list;
  std::multimap<int32_t, int> used_block_list;
  std::unordered_map<std::string, int> tensor_id_map;
  for (auto i = net_defs.begin(); i != net_defs.end(); ++i) {
    const NetDef *net_def = i->second;
    int output_size = net_def->output_info_size();
    for (int i = 0; i < output_size; ++i) {
      const InputOutputInfo &output_info = net_def->output_info(i);
      const auto &output_name = output_info.name();
      auto find_iter = std::find(glb_out_nodes.begin(), glb_out_nodes.end(),
                                 output_name);
      if (find_iter != glb_out_nodes.end()) {  // no need to allocate
        continue;
      }
      const auto &output_dims = output_info.dims();
      auto buf_size = std::accumulate(output_dims.begin(), output_dims.end(),
                                      1, std::multiplies<int32_t>());
      auto bytes = buf_size * GetEnumTypeSize(output_info.data_type());
      auto iter = free_block_list.lower_bound(bytes);
      if (iter == free_block_list.end()) {
        used_block_list.emplace(bytes, i);
        tensor_id_map.emplace(output_name, i);
      } else {
        used_block_list.emplace(iter->first, iter->second);
        tensor_id_map.emplace(output_name, iter->second);
        free_block_list.erase(iter);
      }
    }
    for (auto block : used_block_list) {
      free_block_list.emplace(block.first, block.second);
    }
    used_block_list.clear();
  }

  // allocate memory
  output_tensor_buffers_.resize(free_block_list.size());
  for (auto block : free_block_list) {
    output_tensor_buffers_[block.second] =
        std::shared_ptr<int8_t>(new int8_t[block.first],
                                std::default_delete<int8_t[]>());
  }

  return tensor_id_map;
}

MaceStatus SerialEngine::CreateTensorsForFlows(
    const NetDefMap &net_defs, const std::vector<std::string> &glb_in_nodes,
    const std::vector<std::string> &glb_out_nodes) {
  const auto net_def_size = net_defs.size();
  MACE_CHECK(flows_.size() == net_def_size);

  const auto tensor_id_map = AllocOutTensors(net_defs, glb_out_nodes);

  // Assign the memories to MaceTensors and assign MaceTensors to flows' outputs
  MaceTensorInfo all_out_tensors;
  std::vector<std::string> out_nodes = glb_out_nodes;
  int k = 0;
  for (auto iter = net_defs.begin(); iter != net_defs.end(); ++iter) {
    const NetDef *net_def = iter->second;
    int output_size = net_def->output_info_size();
    auto tensor_info = std::make_shared<MaceTensorInfo>();
    for (int i = 0; i < output_size; ++i) {
      const InputOutputInfo &output_info = net_def->output_info(i);
      const auto &output_name = output_info.name();
      const auto &output_alias = output_info.alias();
      const auto
          &output_key = output_alias.empty() ? output_name : output_alias;
      auto find_iter = std::find(out_nodes.begin(), out_nodes.end(),
                                 output_name);
      if (find_iter != out_nodes.end()) {  // no need to allocate
        out_nodes.erase(find_iter);
        tensor_info->emplace(output_name, MaceTensor());
        all_out_tensors.emplace(output_key, MaceTensor());
        run_helper_.emplace(output_name, tensor_info);
      } else {
        auto idx = tensor_id_map.at(output_name);
        auto output_data = output_tensor_buffers_[idx];
        auto &output_dims = output_info.dims();
        std::vector<int64_t>
            output_shape(output_dims.begin(), output_dims.end());
        auto data_format = static_cast<DataFormat>(output_info.data_format());
        MaceTensor mace_tensor(output_shape, output_data, data_format);
        tensor_info->emplace(output_name, mace_tensor);
        all_out_tensors.emplace(output_key, std::move(mace_tensor));
      }
    }
    output_tensors_.emplace(flows_[k++].get(), tensor_info);
  }
  MACE_CHECK(out_nodes.size() == 0, "can not find output in model: ",
             MakeString(out_nodes));

  // assign MaceTensors to flows' inputs
  std::vector<std::string> in_nodes = glb_in_nodes;
  k = 0;
  for (auto iter = net_defs.begin(); iter != net_defs.end(); ++iter) {
    const NetDef *net_def = iter->second;
    int input_size = net_def->input_info_size();
    auto tensor_info = std::make_shared<MaceTensorInfo>();
    for (int i = 0; i < input_size; ++i) {
      const InputOutputInfo &input_info = net_def->input_info(i);
      const auto &input_name = input_info.name();
      if (all_out_tensors.count(input_name) == 1) {
        tensor_info->emplace(input_name, all_out_tensors.at(input_name));
      } else {
        auto find_iter = std::find(in_nodes.begin(), in_nodes.end(),
                                   input_name);
        MACE_CHECK(find_iter != in_nodes.end(),
                   "Can not find flow's input: ", input_name);
        tensor_info->emplace(input_name, MaceTensor());
        run_helper_.emplace(input_name, tensor_info);
        in_nodes.erase(find_iter);
      }
    }
    input_tensors_.emplace(flows_[k++].get(), tensor_info);
  }
  MACE_CHECK(in_nodes.size() == 0, "can not find input in model: ",
             MakeString(in_nodes));

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus SerialEngine::DoInit(
    const MultiNetDef *multi_net_def,
    const std::vector<std::string> &input_nodes,
    const std::vector<std::string> &output_nodes,
    const unsigned char *model_data, const int64_t model_data_size,
    bool *model_data_unused, BaseEngine *tutor) {
  VLOG(1) << "Initializing SerialEngine";

  // sort the net_def
  NetDefMap net_defs;
  const auto net_def_num = multi_net_def->net_def_size();
  for (int i = 0; i < net_def_num; ++i) {
    const NetDef &net_def = multi_net_def->net_def(i);
    net_defs.emplace(net_def.infer_order(), &net_def);
  }

  // create and init runtimes
  std::unordered_map<const NetDef *, std::shared_ptr<Runtime>> runtime_map;
  MaceStatus ret = CreateAndInitRuntimes(net_defs, &runtime_map, tutor);
  MACE_RETURN_IF_ERROR(ret);

  // create and init flows
  ret = CreateAndInitFlows(net_defs, runtime_map, model_data,
                           model_data_size, model_data_unused);
  MACE_RETURN_IF_ERROR(ret);

  // create flows'output tensors
  ret = CreateTensorsForFlows(net_defs, input_nodes, output_nodes);
  MACE_RETURN_IF_ERROR(ret);

  // check
  auto flow_num = flows_.size();
  auto input_tensor_size = input_tensors_.size();
  auto output_tensor_size = output_tensors_.size();
  MACE_CHECK(input_tensor_size == flow_num && output_tensor_size == flow_num);

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace mace
