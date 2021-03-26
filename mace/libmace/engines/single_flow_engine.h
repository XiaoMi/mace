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

#ifndef MACE_LIBMACE_ENGINES_SINGLE_FLOW_ENGINE_H_
#define MACE_LIBMACE_ENGINES_SINGLE_FLOW_ENGINE_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mace/core/flow/base_flow.h"
#include "mace/libmace/engines/base_engine.h"

namespace mace {
class SingleFlowEngine : public BaseEngine {
 public:
  explicit SingleFlowEngine(const MaceEngineConfig &config);
  ~SingleFlowEngine();

  MaceStatus Init(const MultiNetDef *multi_net_def,
                  const std::vector<std::string> &input_nodes,
                  const std::vector<std::string> &output_nodes,
                  const unsigned char *model_data,
                  const int64_t model_data_size,
                  bool *model_data_unused = nullptr,
                  BaseEngine *tutor = nullptr) override;

  // @Deprecated, will be removed in future version
  MaceStatus Init(const NetDef *net_def,
                  const std::vector<std::string> &input_nodes,
                  const std::vector<std::string> &output_nodes,
                  const unsigned char *model_data,
                  const int64_t model_data_size,
                  bool *model_data_unused) override;

 protected:
  MaceStatus Run(const std::map<std::string, MaceTensor> &inputs,
                 std::map<std::string, MaceTensor> *outputs,
                 RunMetadata *run_metadata) override;

  MaceStatus DoInit(const NetDef *net_def,
                    const std::vector<std::string> &input_nodes,
                    const std::vector<std::string> &output_nodes,
                    const unsigned char *model_data,
                    const int64_t model_data_size,
                    bool *model_data_unused, BaseEngine *tutor);

 private:
  MaceStatus CreateAndInitRuntimes(const NetDef *net_def, BaseEngine *tutor);

 private:
  std::shared_ptr<Runtime> runtime_;
  std::shared_ptr<Runtime> cpu_runtime_;
  std::unique_ptr<BaseFlow> single_flow_;

  MACE_DISABLE_COPY_AND_ASSIGN(SingleFlowEngine);
};

}  // namespace mace

#endif  // MACE_LIBMACE_ENGINES_SINGLE_FLOW_ENGINE_H_
