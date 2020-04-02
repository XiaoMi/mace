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

#include "micro/framework/operator.h"

#include "micro/base/utils.h"
#include "micro/framework/op_context.h"
#include "micro/include/port/define.h"
#include "micro/include/public/micro.h"
#include "micro/model/const_tensor.h"
#include "micro/model/input_output_info.h"
#include "micro/model/net_def.h"
#include "micro/model/operator_def.h"

namespace micro {
namespace framework {

namespace {
const uint16_t kIdxConstTensor = 0xffff;
const uint16_t kIdxModelInput = 0xfffe;
}  // namespace

Operator::~Operator() {}

MaceStatus Operator::Init(MaceMicroEngineConfig *engine_config,
                          framework::OpContext *op_context,
                          const model::OperatorDef *op_def) {
  engine_config_ = engine_config;
  op_context_ = op_context;
  op_def_ = op_def;

  MACE_ASSERT1(op_def_->input_size() == op_context_->input_info_size(),
               "op_def_'s input dosen't match the op_context_'s");
  MACE_ASSERT1(
      op_def_->output_size() == op_context_->output_resize_shape_size(),
      "op_def_'s output dosen't match the op_context_'s");

  return OnInit();
}

MaceStatus Operator::Run() {
  MACE_NOT_IMPLEMENTED;
  return MACE_SUCCESS;
}

MaceStatus Operator::OnInit() {
  return MACE_SUCCESS;
}

const model::Argument *Operator::GetArgByName(const char *name) const {
  MACE_ASSERT(op_def_ != NULL);
  for (uint32_t i = 0; i < op_def_->arg_size(); ++i) {
    const model::Argument *argument = op_def_->arg(i);
    if (base::strcmp(name, argument->name()) == 0) {
      return argument;
    }
  }
  return NULL;
}

uint32_t Operator::GetInputSize() {
  return op_def_->input_size();
}

const void *Operator::DoGetInputData(uint32_t idx) {
  const void *data = NULL;
  const OpIOInfo *input_info = op_context_->input_info(idx);
  const uint32_t op_def_idx = input_info->op_def_idx_;
  if (kIdxConstTensor == op_def_idx) {
    const model::ConstTensor *const_tensor =
        engine_config_->net_def_->tensor(input_info->output_idx_);
    data = engine_config_->model_data_ + const_tensor->offset();
  } else if (kIdxModelInput == op_def_idx) {
    data = engine_config_->input_buffers_[input_info->output_idx_];
  } else {
    const model::OperatorDef *pre_op_def =
        engine_config_->net_def_->op(op_def_idx);
    data = engine_config_->tensor_mem_ +
        pre_op_def->mem_offset(input_info->output_idx_);
  }

  return data;
}

uint32_t Operator::GetInputShapeDimSize(uint32_t idx) {
  uint32_t dim_size = 0;
  const OpIOInfo *input_info = op_context_->input_info(idx);
  const uint32_t op_def_idx = input_info->op_def_idx_;
  if (kIdxConstTensor == op_def_idx) {
    const model::ConstTensor *const_tensor =
        engine_config_->net_def_->tensor(input_info->output_idx_);
    dim_size = const_tensor->dim_size();
  } else if (kIdxModelInput == op_def_idx) {
    const model::InputOutputInfo *info =
        engine_config_->net_def_->input_info(input_info->output_idx_);
    dim_size = info->dim_size();
  } else {
    const model::OperatorDef *op_def = engine_config_->net_def_->op(op_def_idx);
    const model::OutputShape *output_shape =
        op_def->output_shape(input_info->output_idx_);
    dim_size = output_shape->dim_size();
  }
  return dim_size;
}

const int32_t *Operator::GetInputShapeDims(uint32_t idx) {
  const int32_t *dims = NULL;
  const OpIOInfo *input_info = op_context_->input_info(idx);
  const uint32_t op_def_idx = input_info->op_def_idx_;
  if (kIdxConstTensor == op_def_idx) {
    const model::ConstTensor *const_tensor =
        engine_config_->net_def_->tensor(input_info->output_idx_);
    dims = const_tensor->dim();
  } else if (kIdxModelInput == op_def_idx) {
    dims = engine_config_->input_shapes_[input_info->output_idx_];
  } else {
    const model::OperatorDef *op_def = engine_config_->net_def_->op(op_def_idx);
    const model::OutputShape *output_shape =
        op_def->output_shape(input_info->output_idx_);
    dims = output_shape->dim();
  }
  return dims;
}

uint32_t Operator::GetOutputSize() {
  return op_def_->output_size();
}

DataType Operator::GetOutputDataType(uint32_t idx) {
  return op_def_->output_type(idx);
}

void *Operator::DoGetOutputData(uint32_t idx) {
  return engine_config_->tensor_mem_ + op_def_->mem_offset(idx);
}

uint32_t Operator::GetOutputShapeDimSize(uint32_t idx) {
  uint32_t dim_size = 0;
  model::OutputShape *output_shape =
      const_cast<model::OutputShape *>(op_context_->output_resize_shape(idx));
  if (output_shape != NULL) {
    dim_size = output_shape->dim_size();
  }
  return dim_size;
}

const int32_t *Operator::GetOutputShapeDims(uint32_t idx) {
  const int32_t *dims = NULL;
  model::OutputShape *output_shape =
      const_cast<model::OutputShape *>(op_context_->output_resize_shape(idx));
  if (output_shape != NULL) {
    dims = output_shape->dim();
  }
  return dims;
}

MaceStatus Operator::ResizeOutputShape(uint32_t idx, uint32_t dim_size,
                                       const int32_t *dims) {
  model::OutputShape *output_shape =
      const_cast<model::OutputShape *>(op_context_->output_resize_shape(idx));
#ifndef NDEBUG
  if (op_def_->output_shape(idx)->dim_size() < dim_size
      || output_shape->dim_size() < dim_size) {
    LOG(FATAL) << "Can not support dynamic dim_size. op_def_dim_size = "
               << op_def_->output_shape(idx)->dim_size()
               << ", output_shape_dim_size = " << output_shape->dim_size()
               << ", dim_size = " << dim_size;
  }
  int32_t def_output_shape_size =
      base::GetShapeSize(output_shape->dim_size(), output_shape->dim());
  int32_t input_shape_size = base::GetShapeSize(dim_size, dims);
  if (def_output_shape_size < input_shape_size) {
    LOG(INFO) << op_def_->name() << " resize failed, because "
              << def_output_shape_size << " < " << input_shape_size;
    LOG(INFO) << "input: ";
    for (uint32_t i = 0; i < dim_size; ++i) {
      LOG(INFO) << dims[i] << ", ";
    }
    LOG(INFO) << "old output: ";
    for (uint32_t i = 0; i < output_shape->dim_size(); ++i) {
      LOG(INFO) << output_shape->dim(i) << ", ";
    }
    MACE_ASSERT(def_output_shape_size >= input_shape_size);
  }
#endif  // NDEBUG

  if (dim_size > 0) {
    base::memcpy(output_shape->mutable_dim(), dims, dim_size * sizeof(int32_t));
  }
  return MACE_SUCCESS;
}

#ifndef MACE_DEFINE_GET_ARG_BY_NAME_FUNC
#define MACE_DEFINE_GET_ARG_BY_NAME_FUNC(T, FUNC)                   \
template <>                                                         \
T Operator::GetArgByName(const char *name, T default_value) const { \
  const model::Argument *arg = GetArgByName(name);                  \
  if (arg == NULL) {                                                \
    return default_value;                                           \
  } else {                                                          \
    return arg->FUNC();                                             \
  }                                                                 \
}
#endif  // MACE_DEFINE_GET_ARG_BY_NAME_FUNC

MACE_DEFINE_GET_ARG_BY_NAME_FUNC(bool, i)
MACE_DEFINE_GET_ARG_BY_NAME_FUNC(int32_t, i)
MACE_DEFINE_GET_ARG_BY_NAME_FUNC(float, f)

#ifndef MACE_DEFINE_GET_ARRAY_ARG_BY_NAME_FUNC
#define MACE_DEFINE_GET_ARRAY_ARG_BY_NAME_FUNC(T, FUNC)       \
template <>                                                   \
const T *Operator::GetRepeatArgByName(const char *name,       \
                                      uint32_t *size) const { \
  const model::Argument *arg = GetArgByName(name);            \
  if (arg == NULL) {                                          \
    return NULL;                                              \
  }                                                           \
  if (size != NULL) {                                         \
    *size = arg->FUNC##_size();                               \
  }                                                           \
  return arg->FUNC();                                         \
}
#endif  // MACE_DEFINE_GET_ARRAY_ARG_BY_NAME_FUNC

MACE_DEFINE_GET_ARRAY_ARG_BY_NAME_FUNC(int32_t, ints)
MACE_DEFINE_GET_ARRAY_ARG_BY_NAME_FUNC(float, floats)
MACE_DEFINE_GET_ARRAY_ARG_BY_NAME_FUNC(uint8_t, s)

}  // namespace framework
}  // namespace micro
