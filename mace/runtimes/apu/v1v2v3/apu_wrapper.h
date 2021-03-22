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

#ifndef MACE_RUNTIMES_APU_V1V2V3_APU_WRAPPER_H_
#define MACE_RUNTIMES_APU_V1V2V3_APU_WRAPPER_H_

#include <string>
#include <vector>
#include <map>
#include <memory>

#include "mace/core/quantize.h"
#include "mace/core/runtime/runtime.h"
#include "mace/core/tensor.h"
#include "mace/proto/mace.pb.h"

class ApuFrontend;
namespace mace {

class ApuWrapper {
  struct ApuTensorInfo {
    std::string name;
    std::shared_ptr<uint8_t> buf;
    std::vector<index_t> shape;
    int size;
    float scale;
    int zero_point;
    int data_type;
  };

 public:
  explicit ApuWrapper(Runtime *runtime);
  bool Init(const NetDef &net_def, unsigned const char *model_data = nullptr,
            const char *file_name = nullptr,
            bool load = false, bool store = false);
  bool Run(const std::map<std::string, Tensor *> &input_tensors,
           std::map<std::string, Tensor *> *output_tensors);
  bool Uninit();

 protected:
  bool DoInit(const NetDef &net_def, unsigned const char *model_data = nullptr,
              const char *file_name = nullptr,
              bool load = false, bool store = false);

 private:
  int MapToApuDataType(DataType mace_type);
  int MapToApuPoolingMode(int mace_mode);
  int MapToApuEltwiseMode(int mace_mode);
  int GetByteNum(int data_type);

 private:
  ApuFrontend *frontend;
  std::vector<ApuTensorInfo> input_infos_;
  std::vector<ApuTensorInfo> output_infos_;
  QuantizeUtil<float, uint8_t> quantize_util_uint8_;
  QuantizeUtil<float, int16_t> quantize_util_int16_;
};

}  // namespace mace

#endif  // MACE_RUNTIMES_APU_V1V2V3_APU_WRAPPER_H_
