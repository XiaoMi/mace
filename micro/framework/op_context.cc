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

#include "micro/framework/op_context.h"

#include "micro/framework/operator.h"
#include "micro/model/net_def.h"
#include "micro/model/operator_def.h"
#include "micro/include/public/micro.h"

namespace micro {
namespace framework {

MACE_DEFINE_OBJECT_FUNC(OpContext, uint32_t, op_idx)

MACE_DEFINE_PTR_ARRAY_FUNC(OpContext, OpIOInfo, input_info, input_infos_)

MACE_DEFINE_PTR_ARRAY_FUNC(OpContext, model::OutputShape,
                      output_resize_shape, output_resize_shapes_)

MaceStatus OpContext::Init(MaceMicroEngineConfig *engine_config,
                           const model::OperatorDef *op_def) {
  // init OpContext
  uint32_t input_info_size = this->input_info_size();
  for (uint32_t i = 0; i < input_info_size; ++i) {
    Uint2OpIOInfo(this->input_info(i));
  }

  // init Op
  uint32_t op_i = op_idx();
  MACE_RETURN_IF_ERROR(
      engine_config->op_array_[op_i]->Init(engine_config, this, op_def));

  return MACE_SUCCESS;
}

MaceStatus OpContext::Run(MaceMicroEngineConfig *engine_config) {
  return engine_config->op_array_[op_idx()]->Run();
}

}  // namespace framework
}  // namespace micro
