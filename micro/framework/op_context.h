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

#ifndef MICRO_FRAMEWORK_OP_CONTEXT_H_
#define MICRO_FRAMEWORK_OP_CONTEXT_H_

#include "micro/base/serialize.h"
#include "micro/model/operator_def.h"
#include "micro/model/output_shape.h"

namespace micro {

struct MaceMicroEngineConfig;

namespace framework {

class Operator;

class OpContext : public Serialize {
 public:
  MACE_DEFINE_HARD_CODE_MAGIC(OpContext)

  MACE_DECLARE_OBJECT_FUNC(uint32_t, op_idx);
  MACE_DECLARE_PTR_ARRAY_FUNC(OpIOInfo, input_info);
  MACE_DECLARE_PTR_ARRAY_FUNC(model::OutputShape, output_resize_shape);

  MaceStatus Init(MaceMicroEngineConfig *engine_config,
                  const model::OperatorDef *op_def);
  MaceStatus Run(MaceMicroEngineConfig *engine_config);

 protected:
  SerialUint32 op_idx_;
  SerialArray<OpIOInfo> input_infos_;
  SerialArray<model::OutputShape> output_resize_shapes_;
};

}  // namespace framework
}  // namespace micro

#endif  // MICRO_FRAMEWORK_OP_CONTEXT_H_
