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

#include "mace/core/proto/net_def_helper.h"

#include "mace/core/proto/arg_helper.h"

namespace mace {

bool NetDefHelper::HasQuantizedTensor(const NetDef &net_def) {
  for (auto &tensor : net_def.tensors()) {
    if (tensor.quantized()) {
      return true;
    }
  }
  return false;
}

bool NetDefHelper::HasHalfTensor(const NetDef &net_def) {
  for (auto &tensor : net_def.tensors()) {
    if (tensor.data_type() == DataType::DT_HALF) {
      return true;
    }
  }
  return false;
}

index_t NetDefHelper::GetModelValidSize(const NetDef &net_def) {
  index_t valid_data_size = 0;
  for (auto &const_tensor : net_def.tensors()) {
    valid_data_size = std::max<index_t>(
        valid_data_size, const_tensor.offset() +
            const_tensor.data_size()
                * GetEnumTypeSize(const_tensor.data_type()));
  }
  return valid_data_size;
}

bool NetDefHelper::IsQuantizedModel(const NetDef &net_def) {
  int quantize = ProtoArgHelper::GetOptionalArg<NetDef, int>(
      net_def, "quantize_flag", 0);
  return (quantize == 1);
}

}  // namespace mace
