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

#ifndef MACE_LIBMACE_ENGINES_BASE_ENGINE_H_
#define MACE_LIBMACE_ENGINES_BASE_ENGINE_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mace/core/registry/op_delegator_registry.h"
#include "mace/core/registry/ops_registry.h"
#include "mace/core/runtime/runtime.h"
#include "mace/port/file_system.h"
#include "mace/public/mace.h"
#include "mace/utils/macros.h"

namespace mace {

typedef std::unordered_map<uint32_t, std::shared_ptr<Runtime>> RuntimesMap;

class BaseEngine {
 public:
  explicit BaseEngine(const MaceEngineConfig &config);

  virtual ~BaseEngine();

  virtual MaceStatus BeforeInit();
  virtual MaceStatus AfterInit();

  virtual MaceStatus Init(const MultiNetDef *multi_net_def,
                          const std::vector<std::string> &input_nodes,
                          const std::vector<std::string> &output_nodes,
                          const unsigned char *model_data,
                          const int64_t model_data_size,
                          bool *model_data_unused = nullptr,
                          BaseEngine *tutor = nullptr);

  virtual MaceStatus Init(const MultiNetDef *multi_net_def,
                          const std::vector<std::string> &input_nodes,
                          const std::vector<std::string> &output_nodes,
                          const std::string &model_data_file,
                          BaseEngine *tutor = nullptr);

  // @Deprecated, will be removed in future version
  virtual MaceStatus Init(const NetDef *net_def,
                          const std::vector<std::string> &input_nodes,
                          const std::vector<std::string> &output_nodes,
                          const unsigned char *model_data,
                          const int64_t model_data_size,
                          bool *model_data_unused);

  // @Deprecated, will be removed in future version
  virtual MaceStatus Init(const NetDef *net_def,
                          const std::vector<std::string> &input_nodes,
                          const std::vector<std::string> &output_nodes,
                          const std::string &model_data_file);

  virtual MaceStatus Forward(const std::map<std::string, MaceTensor> &inputs,
                             std::map<std::string, MaceTensor> *outputs,
                             RunMetadata *run_metadata);

  virtual MaceStatus ReleaseIntermediateBuffer();
  virtual MaceStatus AllocateIntermediateBuffer();

 protected:
  virtual MaceStatus BeforeRun();
  virtual MaceStatus Run(const std::map<std::string, MaceTensor> &inputs,
                         std::map<std::string, MaceTensor> *outputs,
                         RunMetadata *run_metadata) = 0;
  virtual MaceStatus AfterRun();

  RuntimesMap &GetRuntimesOfTutor(BaseEngine *tutor);

 protected:
  std::unique_ptr<utils::ThreadPool> thread_pool_;
  std::unique_ptr<RuntimeContext> runtime_context_;
  std::unique_ptr<port::ReadOnlyMemoryRegion> model_data_;
  std::unique_ptr<OpRegistry> op_registry_;
  std::unique_ptr<OpDelegatorRegistry> op_delegator_registry_;
  std::shared_ptr<MaceEngineCfgImpl> config_impl_;
  RuntimesMap runtimes_;

  MACE_DISABLE_COPY_AND_ASSIGN(BaseEngine);
};

}  // namespace mace

#endif  // MACE_LIBMACE_ENGINES_BASE_ENGINE_H_
