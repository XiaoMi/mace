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

#include "micro/framework/scratch_buffer.h"
#include "micro/include/utils/macros.h"
#include "micro/ops/substitute_op.h"

namespace micro {
namespace framework {

Operator::~Operator() {}

#ifndef fake_op_
#define fake_op_  (reinterpret_cast<SubstituteOp *>(op_context_))
#endif  // fake_op_

const uint32_t kScratchBufferSize = 100000;
uint8_t kScratchBuffer[kScratchBufferSize] = {0};
MaceMicroEngineConfig kTmpMicroEngineConfig = {
    NULL,  // net_def_;
    NULL,  // model_data_;
    NULL,  // graph_;
    NULL,  // op_array_;
    NULL,  // tensor_mem_;
    NULL,  // input_buffers_;
    NULL,  // input_shapes_;
    kScratchBuffer,
    kScratchBufferSize,
};

MaceStatus Operator::Init(MaceMicroEngineConfig *engine_config,
                          framework::OpContext *op_context,
                          const model::OperatorDef *op_def) {
  engine_config_ = &kTmpMicroEngineConfig;
  op_context_ = op_context;
  MACE_UNUSED(engine_config);
  MACE_UNUSED(op_def_);
  MACE_UNUSED(op_def);

  return OnInit();
}

MaceStatus Operator::OnInit() {
  return MACE_SUCCESS;
}

MaceStatus Operator::Run() {
  MACE_NOT_IMPLEMENTED;
  return MACE_SUCCESS;
}

const model::Argument *Operator::GetArgByName(const char *name) const {
  MACE_UNUSED(name);
  MACE_ASSERT1(false, "Thsi method should not be invoked.");
  return NULL;
}

uint32_t Operator::GetInputSize() {
  return fake_op_->GetInputSize();
}

const void *Operator::DoGetInputData(uint32_t idx) {
  return fake_op_->DoGetInputData(idx);
}

uint32_t Operator::GetInputShapeDimSize(uint32_t idx) {
  return fake_op_->GetInputShapeDimSize(idx);
}

const int32_t *Operator::GetInputShapeDims(uint32_t idx) {
  return fake_op_->GetInputShapeDims(idx);
}

uint32_t Operator::GetOutputSize() {
  return fake_op_->GetOutputSize();
}

void *Operator::DoGetOutputData(uint32_t idx) {
  return fake_op_->DoGetOutputData(idx);
}

uint32_t Operator::GetOutputShapeDimSize(uint32_t idx) {
  return fake_op_->GetOutputShapeDimSize(idx);
}

const int32_t *Operator::GetOutputShapeDims(uint32_t idx) {
  return fake_op_->GetOutputShapeDims(idx);
}

MaceStatus Operator::ResizeOutputShape(uint32_t idx, uint32_t dim_size,
                                       const int32_t *dims) {
  return fake_op_->ResizeOutputShape(idx, dim_size, dims);
}

#ifndef MACE_DEFINE_GET_ARG_BY_NAME_FUNC
#define MACE_DEFINE_GET_ARG_BY_NAME_FUNC(T, FUNC)                   \
template <>                                                         \
T Operator::GetArgByName(const char *name, T default_value) const { \
  return fake_op_->GetArgByName<T>(name, default_value);            \
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
  return fake_op_->GetRepeatArgByName<T>(name, size);         \
}
#endif  // MACE_DEFINE_GET_ARRAY_ARG_BY_NAME_FUNC

MACE_DEFINE_GET_ARRAY_ARG_BY_NAME_FUNC(int32_t, ints)
MACE_DEFINE_GET_ARRAY_ARG_BY_NAME_FUNC(float, floats)
MACE_DEFINE_GET_ARRAY_ARG_BY_NAME_FUNC(uint8_t, s)

}  // namespace framework
}  // namespace micro
