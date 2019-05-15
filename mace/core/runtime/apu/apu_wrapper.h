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

#ifndef MACE_CORE_RUNTIME_APU_APU_WRAPPER_H_
#define MACE_CORE_RUNTIME_APU_APU_WRAPPER_H_

#include <string>
#include <vector>
#include <map>
#include <memory>

#include "mace/proto/mace.pb.h"
#include "mace/core/tensor.h"
#include "mace/core/device.h"
#include "mace/core/quantize.h"

#include "third_party/apu/ApuFrontend.h"

namespace mace {

class ApuWrapper {
struct tensor_info {
    std::string name;
    std::shared_ptr<uint8_t> buf;
    std::vector<index_t> shape;
    int size;
    float scale;
    int zero_point;
};

 public:
    explicit ApuWrapper(Device *device);
    bool Init(const NetDef& net_def, unsigned const char *model_data);
    bool Run(const std::map<std::string, Tensor *> &input_tensors,
             std::map<std::string, Tensor *> *output_tensors);
    bool Uninit();

 private:
    apu_data_type MapToApuDataType(DataType mace_type);
    apu_pooling_mode MapToApuPoolingMode(int mace_mode);
    apu_eltwise_mode MapToApuEltwiseMode(int mace_mode);

 private:
    ApuFrontend* frontend;
    std::vector<tensor_info> input_infos;
    std::vector<tensor_info> output_infos;
    QuantizeUtil<uint8_t> quantize_util_;
};

}  // namespace mace

#endif  // MACE_CORE_RUNTIME_APU_APU_WRAPPER_H_
