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

#ifndef MACE_CORE_OPS_OPS_UTILS_H_
#define MACE_CORE_OPS_OPS_UTILS_H_

#include <string>
#include <vector>

#include "mace/core/types.h"
#include "mace/port/port.h"
#include "mace/proto/mace.pb.h"

namespace mace {

enum BufferContentType {
  // for opencl
  CONV2D_FILTER = 0,
  IN_OUT_CHANNEL = 1,
  ARGUMENT = 2,
  IN_OUT_HEIGHT = 3,
  IN_OUT_WIDTH = 4,
  WINOGRAD_FILTER = 5,
  DW_CONV2D_FILTER = 6,
  WEIGHT_HEIGHT = 7,
  WEIGHT_WIDTH = 8,
};

class OpConditionContext;
class OpsUtils {
 public:
  static void BuildTransformOpDef(
      const std::string &input_name,
      const std::vector<index_t> &input_shape,
      const std::string &output_name,
      const DataType dt,
      const BufferContentType buffer_type,
      const MemoryType mem_type,
      DataFormat data_format,
      OperatorDef *op_def);
};

}  // namespace mace

#endif  // MACE_CORE_OPS_OPS_UTILS_H_
