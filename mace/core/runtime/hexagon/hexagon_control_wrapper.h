// Copyright 2018 Xiaomi, Inc.  All rights reserved.
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

#ifndef MACE_CORE_RUNTIME_HEXAGON_HEXAGON_CONTROL_WRAPPER_H_
#define MACE_CORE_RUNTIME_HEXAGON_HEXAGON_CONTROL_WRAPPER_H_

#include <vector>

#include "mace/core/runtime/hexagon/quantize.h"
#include "mace/core/tensor.h"
#include "mace/public/mace.h"
#include "third_party/nnlib/hexagon_nn.h"

namespace mace {

class HexagonControlWrapper {
 public:
  HexagonControlWrapper() {}
  int GetVersion();
  bool Config();
  bool Init();
  bool Finalize();
  bool SetupGraph(const NetDef &net_def, const unsigned char *model_data);
  bool ExecuteGraph(const Tensor &input_tensor, Tensor *output_tensor);
  bool ExecuteGraphNew(const std::vector<Tensor> &input_tensors,
                       std::vector<Tensor> *output_tensors);
  bool ExecuteGraphPreQuantize(const Tensor &input_tensor,
                               Tensor *output_tensor);

  bool TeardownGraph();
  void PrintLog();
  void PrintGraph();
  void GetPerfInfo();
  void ResetPerfInfo();
  void SetDebugLevel(int level);

 private:
  static constexpr int NODE_ID_OFFSET = 10000;

  inline uint32_t node_id(uint32_t nodeid) { return NODE_ID_OFFSET + nodeid; }

  int nn_id_;
  Quantizer quantizer_;

  std::vector<std::vector<index_t>> input_shapes_;
  std::vector<std::vector<index_t>> output_shapes_;
  std::vector<DataType> input_data_types_;
  std::vector<DataType> output_data_types_;
  uint32_t num_inputs_;
  uint32_t num_outputs_;

  MACE_DISABLE_COPY_AND_ASSIGN(HexagonControlWrapper);
};
}  // namespace mace

#endif  // MACE_CORE_RUNTIME_HEXAGON_HEXAGON_CONTROL_WRAPPER_H_
