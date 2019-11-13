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

#include "micro/framework/graph.h"

#include "micro/base/logging.h"
#include "micro/base/serialize.h"
#include "micro/base/utils.h"
#include "micro/framework/operator.h"
#include "micro/include/public/micro.h"
#include "micro/model/net_def.h"

namespace micro {
namespace framework {

MACE_DEFINE_PTR_ARRAY_FUNC(Graph, OpContext, op_context, op_contexts_)
MACE_DEFINE_ARRAY_FUNC(Graph, uint32_t, input_op_idx, input_op_idxs_);
MACE_DEFINE_PTR_ARRAY_FUNC(Graph, OpIOInfo, output_info, output_infos_);

MaceStatus Graph::Init(MaceMicroEngineConfig *engine_config) {
  MACE_ASSERT(engine_config->net_def_->op_size() == op_context_size());

  uint32_t output_info_size = this->output_info_size();
  for (uint32_t i = 0; i < output_info_size; ++i) {
    Uint2OpIOInfo(this->output_info(i));
  }

  uint32_t op_size = engine_config->net_def_->op_size();
  for (uint32_t i = 0; i < op_size; ++i) {
    OpContext *op_ctx = const_cast<OpContext *>(op_context(i));
    MACE_RETURN_IF_ERROR(op_ctx->Init(
        engine_config, engine_config->net_def_->op(i)));
  }

  return MACE_SUCCESS;
}

MaceStatus Graph::RegisterInputData(MaceMicroEngineConfig *engine_config,
                                    uint32_t idx,
                                    const void *input_buffer,
                                    const int32_t *input_dims) {
  engine_config->input_buffers_[idx] = input_buffer;
  engine_config->input_shapes_[idx] = input_dims;

  // update the op's input buffers
  uint32_t op_idx = input_op_idx(idx);
  framework::Operator *input_op = engine_config->op_array_[op_idx];
  return input_op->OnInit();
}

MaceStatus Graph::Run(MaceMicroEngineConfig *engine_config) {
  uint32_t op_size = engine_config->net_def_->op_size();
  for (uint32_t i = 0; i < op_size; ++i) {
    OpContext *op_ctx = const_cast<OpContext *>(op_context(i));
    MACE_RETURN_IF_ERROR(op_ctx->Run(engine_config));
  }

  return MACE_SUCCESS;
}

MaceStatus Graph::GetOutputData(MaceMicroEngineConfig *engine_config,
                                const uint32_t idx,
                                void **output_data,
                                const int32_t **output_dims,
                                uint32_t *output_dim_size) {
  MACE_ASSERT(idx < output_info_size());

  const OpIOInfo *o_info = output_info(idx);
  return GetOpOutputData(engine_config, o_info->op_def_idx_,
                         o_info->output_idx_, output_data,
                         output_dims, output_dim_size);
}

MaceStatus Graph::GetOpOutputData(MaceMicroEngineConfig *engine_config,
                                  const uint32_t op_def_idx,
                                  const uint32_t output_idx,
                                  void **output_data,
                                  const int32_t **output_dims,
                                  uint32_t *output_dim_size) {
  MACE_ASSERT(engine_config != NULL);
  MACE_ASSERT(output_data != NULL);
  MACE_ASSERT(output_dims != NULL);
  MACE_ASSERT(output_dim_size != NULL);

  const model::OperatorDef *op_def = engine_config->net_def_->op(op_def_idx);
  *output_data = engine_config->tensor_mem_ + op_def->mem_offset(output_idx);

  const model::OutputShape *output_shape =
      op_context(op_def_idx)->output_resize_shape(output_idx);
  *output_dims = output_shape->dim();
  *output_dim_size = output_shape->dim_size();

  return MACE_SUCCESS;
}

}  // namespace framework
}  // namespace micro
