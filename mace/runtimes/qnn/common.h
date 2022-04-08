// Copyright 2021 The MACE Authors. All Rights Reserved.
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

#ifndef MACE_RUNTIMES_QNN_COMMON_H_
#define MACE_RUNTIMES_QNN_COMMON_H_

#include <memory>
#include <string>
#include <vector>
#include <utility>

#include "mace/core/tensor.h"
#include "QnnInterface.h"

namespace mace {
enum QnnGraphState {
  QNN_INIT_START = 0,
  QNN_INIT_DONE = 1,
  QNN_INFERENCE_START = 2,
  QNN_INFERENCE_DONE = 3
};

struct QnnInOutInfo {
  QnnInOutInfo(const std::string &name,
               const DataType data_type,
               const std::vector<uint32_t> &shape,
               const float scale,
               const int32_t zero_point,
               std::unique_ptr<Tensor> quantized_tensor)
      :  name(name),
         data_type(data_type),
         shape(shape),
         scale(scale),
         zero_point(zero_point),
         quantized_tensor(std::move(quantized_tensor)) {}

  std::string name;
  DataType data_type;
  std::vector<uint32_t> shape;
  float scale;
  int32_t zero_point;
  std::unique_ptr<Tensor> quantized_tensor;
};

enum class StatusCode {
  SUCCESS,
  FAILURE,
  FAIL_LOAD_BACKEND,
  FAIL_LOAD_MODEL,
  FAIL_SYM_FUNCTION,
  FAIL_GET_INTERFACE_PROVIDERS,
  FAIL_LOAD_SYSTEM_LIB,
};

struct QnnFunctionPointers {
  QNN_INTERFACE_VER_TYPE qnnInterface;
};

}  // namespace mace

#endif  // MACE_RUNTIMES_QNN_COMMON_H_
