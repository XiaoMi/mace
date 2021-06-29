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

#ifndef MACE_LIBMACE_ENGINES_SERIAL_ENGINE_H_
#define MACE_LIBMACE_ENGINES_SERIAL_ENGINE_H_

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "mace/core/flow/base_flow.h"
#include "mace/libmace/engines/base_engine.h"
#include "mace/public/mace.h"

namespace mace {
class SerialEngine : public BaseEngine {
 public:
  explicit SerialEngine(const MaceEngineConfig &config);

  ~SerialEngine();

  MaceStatus Init(const MultiNetDef *net_def,
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
                  bool *model_data_unused = nullptr) override;

  MaceStatus ReleaseIntermediateBuffer() override;
  MaceStatus AllocateIntermediateBuffer() override;

 protected:
  MaceStatus BeforeRun() override;
  MaceStatus Run(const std::map<std::string, MaceTensor> &inputs,
                 std::map<std::string, MaceTensor> *outputs,
                 RunMetadata *run_metadata) override;
  MaceStatus AfterRun() override;
  MaceStatus FakeWarmup() override;

 private:
  typedef std::unordered_map<const NetDef *,
                             std::shared_ptr<Runtime>> NetRuntimeMap;
  typedef std::map<std::string, MaceTensor> MaceTensorInfo;
  typedef std::unordered_map<const BaseFlow *,
                             std::shared_ptr<MaceTensorInfo>> FlowTensorMap;
  typedef std::vector<std::unique_ptr<BaseFlow>> FlowArray;
  typedef std::map<int, const NetDef *> NetDefMap;
  MaceStatus DoInit(const MultiNetDef *multi_net_def,
                    const std::vector<std::string> &input_nodes,
                    const std::vector<std::string> &output_nodes,
                    const unsigned char *model_data,
                    const int64_t model_data_size,
                    bool *model_data_unused, BaseEngine *tutor);

  MaceStatus CreateAndInitRuntimes(const NetDefMap &net_defs,
                                   NetRuntimeMap *runtime_map,
                                   BaseEngine *tutor);

  MaceStatus CreateAndInitFlows(
      const NetDefMap &net_defs, const NetRuntimeMap &runtime_map,
      const unsigned char *model_data, const int64_t model_data_size,
      bool *model_data_unused);

  std::unordered_map<std::string, int> AllocOutTensors(
      const NetDefMap &net_defs, const std::vector<std::string> &glb_out_nodes);

  MaceStatus CreateTensorsForFlows(
      const NetDefMap &net_defs, const std::vector<std::string> &input_nodes,
      const std::vector<std::string> &output_nodes);

 private:
  std::shared_ptr<Runtime> cpu_runtime_;
  FlowArray flows_;

  FlowTensorMap input_tensors_;
  FlowTensorMap output_tensors_;
  std::vector<std::shared_ptr<void>> output_tensor_buffers_;
  std::unordered_map<std::string, std::shared_ptr<MaceTensorInfo>> run_helper_;

  bool inter_mem_released_;

  MACE_DISABLE_COPY_AND_ASSIGN(SerialEngine);
};

}  // namespace mace

#endif  // MACE_LIBMACE_ENGINES_SERIAL_ENGINE_H_
