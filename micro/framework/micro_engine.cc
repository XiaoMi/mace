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
#include "micro/framework/operator.h"

namespace micro {

MaceMicroEngine::MaceMicroEngine() : engine_config_(nullptr),
                                     initialized_(false) {}

MaceStatus MaceMicroEngine::Init(MaceMicroEngineConfig *engine_config) {
  MACE_ASSERT(engine_config != NULL && engine_config->net_def_ != NULL
                  && engine_config->model_data_ != NULL
                  && engine_config->graph_ != NULL
                  && engine_config->op_array_ != NULL
                  && engine_config->tensor_mem_ != NULL);
  if (initialized_) {
    MACE_ASSERT1(engine_config == engine_config_,
                 "The engine has initialized and get an invalid config.");
    return MACE_SUCCESS;
  }
  engine_config_ = engine_config;

  MACE_RETURN_IF_ERROR(engine_config_->graph_->Init(engine_config_));

  initialized_ = true;
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
  MACE_ASSERT1(initialized_ == true, "The engine has not initialized.");

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

MaceMicroEngineConfig *MaceMicroEngine::GetEngineConfig() {
  return engine_config_;
}

MaceStatus CreateMaceMicroEngineFromBinary(uint8_t *model_data,
                                           uint32_t size,
                                           framework::Operator **op_array,
                                           uint32_t input_num,
                                           MaceMicroEngine **engine) {
  struct model_header {
    int64_t net_def_offset;
    int64_t graph_offset;
    int64_t model_offset;
    int64_t model_end;
    int64_t tensor_mem_size;
    int64_t scratch_buffer_size;
  };

  auto header = reinterpret_cast<model_header *>(model_data);
  MACE_ASSERT(static_cast<int64_t>(size) == header->model_end);
  auto net_def = reinterpret_cast<micro::model::NetDef *>(
      model_data + header->net_def_offset);
  auto graph = reinterpret_cast<micro::framework::Graph *>(
      model_data + header->graph_offset);
  auto model = reinterpret_cast<uint8_t *>(model_data + header->model_offset);

  auto tensor_mem = new uint8_t[header->tensor_mem_size];
  auto scratch_buffer = new uint8_t[header->scratch_buffer_size];

  const void **input_buffers = new const void *[input_num];
  const int32_t **input_shapes = new const int32_t *[input_num];
  for (uint32_t i = 0; i < input_num; ++i) {
    input_buffers[i] = nullptr;
    input_shapes[i] = nullptr;
  }

  *engine = new micro::MaceMicroEngine();
  micro::MaceMicroEngineConfig *engine_config =
      new micro::MaceMicroEngineConfig{net_def,
                                       model,
                                       graph,
                                       op_array,
                                       tensor_mem,
                                       input_buffers,
                                       input_shapes,
                                       scratch_buffer,
                                       static_cast<uint32_t>(header->scratch_buffer_size)};
  return (*engine)->Init(engine_config);
}

void DestroyMicroEngineFromBinary(micro::MaceMicroEngine *engine) {
  delete[] engine->GetEngineConfig()->tensor_mem_;
  delete[] engine->GetEngineConfig()->scratch_buffer_;
  delete[] engine->GetEngineConfig()->input_buffers_;
  delete[] engine->GetEngineConfig()->input_shapes_;
  delete engine->GetEngineConfig();
  delete engine;
}

}  // namespace micro
