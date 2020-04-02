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


#ifndef MICRO_FRAMEWORK_GRAPH_H_
#define MICRO_FRAMEWORK_GRAPH_H_

#include "micro/base/serialize.h"
#include "micro/framework/op_context.h"

namespace micro {

struct MaceMicroEngineConfig;

namespace framework {

class Graph : public Serialize {
 public:
  MACE_DEFINE_HARD_CODE_MAGIC(Graph)

  MACE_DECLARE_PTR_ARRAY_FUNC(OpContext, op_context);
  MACE_DECLARE_ARRAY_FUNC(uint32_t, input_op_idx);
  MACE_DECLARE_PTR_ARRAY_FUNC(OpIOInfo, output_info);

  MaceStatus Init(MaceMicroEngineConfig *engine_config);
  MaceStatus RegisterInputData(MaceMicroEngineConfig *engine_config,
                               uint32_t idx,
                               const void *input_buffer,
                               const int32_t *input_dims);
  MaceStatus Run(MaceMicroEngineConfig *engine_config);
  MaceStatus GetOutputData(MaceMicroEngineConfig *engine_config,
                           const uint32_t idx,
                           void **output_data,
                           const int32_t **output_dims,
                           uint32_t *output_dim_size);
  MaceStatus GetOpOutputData(MaceMicroEngineConfig *engine_config,
                             const uint32_t op_def_idx,
                             const uint32_t output_idx,
                             void **output_data,
                             const int32_t **output_dims,
                             uint32_t *output_dim_size);

 protected:
  SerialArray<OpContext> op_contexts_;
  SerialArray<SerialUint32> input_op_idxs_;
  SerialArray<OpIOInfo> output_infos_;
};

}  // namespace framework
}  // namespace micro

#endif  // MICRO_FRAMEWORK_GRAPH_H_
