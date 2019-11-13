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

#include "micro/base/logging.h"
#include "micro/base/utils.h"
#include "micro/framework/graph.h"
#include "micro/framework/scratch_buffer.h"
#include "micro/include/public/micro.h"
#include "micro/model/net_def.h"
#include "micro/model/operator_def.h"
#include "micro/port/api.h"

namespace micro {
MaceStatus MaceMicroEngine::Init(MaceMicroEngineConfig *engine_config) {
  MACE_ASSERT(engine_config != NULL && engine_config->net_def_ != NULL
                  && engine_config->model_data_ != NULL
                  && engine_config->graph_ != NULL
                  && engine_config->op_array_ != NULL
                  && engine_config->tensor_mem_ != NULL);
  engine_config_ = engine_config;

  MACE_RETURN_IF_ERROR(engine_config_->graph_->Init(engine_config_));

  return MACE_SUCCESS;
}

MaceStatus MaceMicroEngine::RegisterInputData(uint32_t idx,
                                              const void *input_buffer,
                                              const int32_t *input_dims) {
  MACE_ASSERT(idx < engine_config_->net_def_->input_info_size());
  MACE_ASSERT(input_buffer != NULL);
  MACE_ASSERT(input_dims != NULL);

  return engine_config_->graph_->RegisterInputData(engine_config_, idx,
                                                   input_buffer, input_dims);
}

MaceStatus MaceMicroEngine::Run() {
  return engine_config_->graph_->Run(engine_config_);
}

MaceStatus MaceMicroEngine::GetOutputData(const uint32_t idx,
                                          void **output_data,
                                          const int32_t **output_dims,
                                          uint32_t *output_dim_size) {
  return engine_config_->graph_->GetOutputData(engine_config_, idx,
                                               output_data, output_dims,
                                               output_dim_size);
}

MaceStatus MaceMicroEngine::GetOpOutputData(const uint32_t op_def_idx,
                                            const uint32_t output_idx,
                                            void **output_data,
                                            const int32_t **output_dims,
                                            uint32_t *output_dim_size) {
  return engine_config_->graph_->GetOpOutputData(engine_config_, op_def_idx,
                                                 output_idx, output_data,
                                                 output_dims, output_dim_size);
}

MaceMicroEngine::MaceMicroEngine(const MaceMicroEngine &) {
  MACE_NOT_IMPLEMENTED;
}

MaceMicroEngine &MaceMicroEngine::operator=(const MaceMicroEngine &) {
  MACE_NOT_IMPLEMENTED;
  return *this;
}

}  // namespace micro
