// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mace/core/tensor.h"
#include "mace/public/mace.h"

namespace mace {

struct InOutInfo {
  InOutInfo(const std::string &name,
            const std::vector<index_t> &shape,
            const DataType data_type,
            const float scale,
            const int32_t zero_point,
            std::unique_ptr<Tensor> tensor_u8)
      :  name(name),
         shape(shape),
         data_type(data_type),
         scale(scale),
         zero_point(zero_point),
         tensor_u8(std::move(tensor_u8)) {}

  std::string name;
  std::vector<index_t> shape;
  DataType data_type;
  float scale;
  int32_t zero_point;
  std::unique_ptr<Tensor> tensor_u8;
};

class HexagonControlWrapper {
 public:
  HexagonControlWrapper() = default;
  virtual ~HexagonControlWrapper() = default;

  virtual int GetVersion() = 0;
  virtual bool Config() = 0;
  virtual bool Init() = 0;
  virtual bool Finalize() = 0;
  virtual bool SetupGraph(const NetDef &net_def,
                          const unsigned char *model_data) = 0;
  virtual bool ExecuteGraph(const Tensor &input_tensor,
                            Tensor *output_tensor) = 0;
  virtual bool ExecuteGraphNew(
      const std::map<std::string, Tensor*> &input_tensors,
      std::map<std::string, Tensor*> *output_tensors) = 0;
  virtual bool TeardownGraph() = 0;
  virtual void PrintLog() = 0;
  virtual void PrintGraph() = 0;
  virtual void GetPerfInfo() = 0;
  virtual void ResetPerfInfo() = 0;
  virtual void SetDebugLevel(int level) = 0;

 protected:
  static constexpr int kNodeIdOffset = 10000;
  static constexpr int kNumMetaData = 4;

  inline uint32_t node_id(uint32_t nodeid) { return kNodeIdOffset + nodeid; }

  int nn_id_;

  std::vector<InOutInfo> input_info_;
  std::vector<InOutInfo> output_info_;
  int num_inputs_;
  int num_outputs_;

  MACE_DISABLE_COPY_AND_ASSIGN(HexagonControlWrapper);
};
}  // namespace mace

#endif  // MACE_CORE_RUNTIME_HEXAGON_HEXAGON_CONTROL_WRAPPER_H_
