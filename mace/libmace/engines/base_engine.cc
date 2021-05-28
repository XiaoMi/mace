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


#include "mace/libmace/engines/base_engine.h"

#include <algorithm>
#include <numeric>
#include <memory>
#include <set>
#include <utility>

#include "mace/core/flow/base_flow.h"
#include "mace/core/flow/flow_registry.h"
#include "mace/core/memory/rpcmem/rpcmem.h"
#include "mace/core/registry/ops_registry.h"
#include "mace/core/registry/op_delegator_registry.h"
#include "mace/core/runtime/runtime_context.h"
#include "mace/core/runtime/runtime_registry.h"
#include "mace/core/runtime/runtime.h"
#include "mace/ops/registry/registry.h"
#include "mace/utils/mace_engine_config.h"
#include "mace/utils/memory.h"
#include "mace/port/env.h"
#include "mace/port/file_system.h"

namespace mace {

BaseEngine::BaseEngine(const MaceEngineConfig &config)
    : thread_pool_(new utils::ThreadPool(config.impl_->num_threads(),
                                         config.impl_->cpu_affinity_policy())),
      model_data_(nullptr), op_registry_(new OpRegistry),
      op_delegator_registry_(new OpDelegatorRegistry),
      config_impl_(config.impl_) {
#ifdef MACE_ENABLE_RPCMEM
  runtime_context_ = make_unique<IonRuntimeContext>(
      thread_pool_.get(), rpcmem_factory::CreateRpcmem());
#else
  runtime_context_ = make_unique<RuntimeContext>(thread_pool_.get());
#endif  // MACE_ENABLE_RPCMEM
}

MaceStatus BaseEngine::Init(
    const MultiNetDef *multi_net_def,
    const std::vector<std::string> &input_nodes,
    const std::vector<std::string> &output_nodes,
    const unsigned char *model_data, const int64_t model_data_size,
    bool *model_data_unused, BaseEngine *tutor) {
  thread_pool_->Init();

  // register ops and delegators
  ops::RegisterAllOps(op_registry_.get());
  ops::RegisterAllOpDelegators(op_delegator_registry_.get());

  MACE_UNUSED(multi_net_def);
  MACE_UNUSED(input_nodes);
  MACE_UNUSED(output_nodes);
  MACE_UNUSED(model_data);
  MACE_UNUSED(model_data_size);
  MACE_UNUSED(model_data_unused);
  MACE_UNUSED(tutor);

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus BaseEngine::Init(
    const MultiNetDef *multi_net_def,
    const std::vector<std::string> &input_nodes,
    const std::vector<std::string> &output_nodes,
    const std::string &model_data_file, BaseEngine *tutor) {
  VLOG(3) << "Loading Model Data";

  auto fs = GetFileSystem();
  MACE_RETURN_IF_ERROR(fs->NewReadOnlyMemoryRegionFromFile(
      model_data_file.c_str(), &model_data_));

  bool model_data_unused = false;
  MACE_RETURN_IF_ERROR(Init(
      multi_net_def, input_nodes, output_nodes,
      reinterpret_cast<const unsigned char *>(model_data_->data()),
      model_data_->length(), &model_data_unused, tutor));

  if (model_data_unused) {
    model_data_.reset();
  }

  return MaceStatus::MACE_SUCCESS;
}

// @Deprecated, will be removed in future version
MaceStatus BaseEngine::Init(
    const NetDef *net_def, const std::vector<std::string> &input_nodes,
    const std::vector<std::string> &output_nodes,
    const unsigned char *model_data, const int64_t model_data_size,
    bool *model_data_unused) {
  thread_pool_->Init();
  // register ops and delegators
  ops::RegisterAllOps(op_registry_.get());
  ops::RegisterAllOpDelegators(op_delegator_registry_.get());

  MACE_UNUSED(net_def);
  MACE_UNUSED(input_nodes);
  MACE_UNUSED(output_nodes);
  MACE_UNUSED(model_data);
  MACE_UNUSED(model_data_size);
  MACE_UNUSED(model_data_unused);

  return MaceStatus::MACE_SUCCESS;
}

// @Deprecated, will be removed in future version
MaceStatus BaseEngine::Init(
    const NetDef *net_def,
    const std::vector<std::string> &input_nodes,
    const std::vector<std::string> &output_nodes,
    const std::string &model_data_file) {
  VLOG(3) << "Loading Model Data";

  auto fs = GetFileSystem();
  MACE_RETURN_IF_ERROR(fs->NewReadOnlyMemoryRegionFromFile(
      model_data_file.c_str(), &model_data_));

  bool model_data_unused = false;
  MACE_RETURN_IF_ERROR(Init(
      net_def, input_nodes, output_nodes,
      reinterpret_cast<const unsigned char *>(model_data_->data()),
      model_data_->length(), &model_data_unused));

  if (model_data_unused) {
    model_data_.reset();
  }

  return MaceStatus::MACE_SUCCESS;
}


MaceStatus BaseEngine::BeforeInit() {
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus BaseEngine::AfterInit() {
  for (auto i = runtimes_.begin(); i != runtimes_.end(); ++i) {
    // Release the intermediate buffer for the other engines' reuse
    i->second->ReleaseAllBuffer(RENT_SHARE, false);
  }
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus BaseEngine::ReleaseIntermediateBuffer() {
  MACE_NOT_IMPLEMENTED;
  return MaceStatus::MACE_UNSUPPORTED;
}

MaceStatus BaseEngine::AllocateIntermediateBuffer() {
  MACE_NOT_IMPLEMENTED;
  return MaceStatus::MACE_UNSUPPORTED;
}

RuntimesMap &BaseEngine::GetRuntimesOfTutor(BaseEngine *tutor) {
  MACE_CHECK(!tutor->runtimes_.empty(),
             "Before using the tutor engine, you must init it.");

  return tutor->runtimes_;
}

std::vector<RuntimeType> BaseEngine::GetRuntimeTypes() {
  std::set<RuntimeType> runtime_types;
  for (auto &runtime : runtimes_) {
    runtime_types.insert(static_cast<RuntimeType>(runtime.first >> 16));
  }
  return std::vector<RuntimeType>(runtime_types.begin(), runtime_types.end());
}

MaceStatus BaseEngine::Forward(const std::map<std::string, MaceTensor> &inputs,
                               std::map<std::string, MaceTensor> *outputs,
                               RunMetadata *run_metadata) {
  MACE_RETURN_IF_ERROR(BeforeRun());
  MACE_RETURN_IF_ERROR(Run(inputs, outputs, run_metadata));
  return AfterRun();
}

MaceStatus BaseEngine::BeforeRun() {
  for (auto i = runtimes_.begin(); i != runtimes_.end(); ++i) {
    MACE_RETURN_IF_ERROR(i->second->BeforeRun(config_impl_.get()));
  }
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus BaseEngine::AfterRun() {
  for (auto i = runtimes_.begin(); i != runtimes_.end(); ++i) {
    MACE_RETURN_IF_ERROR(i->second->AfterRun());
  }
  return MaceStatus::MACE_SUCCESS;
}

BaseEngine::~BaseEngine() {}

}  // namespace mace
