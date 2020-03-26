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

#ifndef MACE_CORE_RUNTIME_HEXAGON_HEXAGON_HTA_WRAPPER_H_
#define MACE_CORE_RUNTIME_HEXAGON_HEXAGON_HTA_WRAPPER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>

#include "mace/core/device.h"
#include "mace/core/runtime/hexagon/hexagon_control_wrapper.h"
#include "mace/core/runtime/hexagon/hexagon_hta_transformer.h"
#include "mace/core/tensor.h"
#include "mace/public/mace.h"
#include "third_party/hta/hta_hexagon_api.h"

namespace mace {

struct HTAInOutInfo : public InOutInfo {
  HTAInOutInfo(const std::string &name,
               const std::vector<index_t> &shape,
               const DataType data_type,
               const float scale,
               const int32_t zero_point,
               std::unique_ptr<Tensor> quantized_tensor,
               std::unique_ptr<Tensor> hta_tensor)
      :  InOutInfo(name, shape, data_type),
         scale(scale),
         zero_point(zero_point),
         quantized_tensor(std::move(quantized_tensor)),
         hta_tensor(std::move(hta_tensor)) {}

  float scale;
  int32_t zero_point;
  std::unique_ptr<Tensor> quantized_tensor;
  std::unique_ptr<Tensor> hta_tensor;
};

class HexagonHTAWrapper : public HexagonControlWrapper {
 public:
  explicit HexagonHTAWrapper(Device *device);

  int GetVersion() override;
  bool Config() override;
  bool Init() override;
  bool Finalize() override;
  bool SetupGraph(const NetDef &net_def,
                  const unsigned char *model_data) override;
  bool ExecuteGraph(const Tensor &input_tensor,
                    Tensor *output_tensor) override;
  bool ExecuteGraphNew(const std::map<std::string, Tensor*> &input_tensors,
                       std::map<std::string, Tensor*> *output_tensors) override;
  bool TeardownGraph() override;
  void PrintLog() override;
  void PrintGraph() override;
  void GetPerfInfo() override;
  void ResetPerfInfo() override;
  void SetDebugLevel(int level) override;

 private:
  Allocator *allocator_;
  std::vector<HTAInOutInfo> input_info_;
  std::vector<HTAInOutInfo> output_info_;
  std::vector<hexagon_hta_nn_hw_tensordef> input_tensordef_;
  std::vector<hexagon_hta_nn_hw_tensordef> output_tensordef_;
  std::unique_ptr<HexagonHTATranformerBase> transformer_;
  MACE_DISABLE_COPY_AND_ASSIGN(HexagonHTAWrapper);
};
}  // namespace mace

#endif  // MACE_CORE_RUNTIME_HEXAGON_HEXAGON_HTA_WRAPPER_H_
