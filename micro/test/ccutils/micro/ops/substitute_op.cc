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
#include "micro/include/public/micro.h"
#include "micro/include/utils/macros.h"
#include "micro/ops/substitute_op.h"

namespace micro {
namespace framework {

SubstituteOp::SubstituteOp()
    : input_idx_(0), output_idx_(0), arg_idx_(0), repeat_arg_idx_(0) {}

SubstituteOp &SubstituteOp::AddInput(
    const void *input, const int32_t *dims, const uint32_t dims_size) {
  MACE_ASSERT1(input != NULL || dims != NULL || dims_size == 0,
               "Invalid param");
  MACE_ASSERT1(input_idx_ < kMaxInputNum, "Not enough mem.");
  inputs_[input_idx_] = input;
  input_dims_[input_idx_] = dims;
  input_dim_sizes_[input_idx_] = dims_size;
  ++input_idx_;
  return *this;
}

SubstituteOp &SubstituteOp::AddOutput(
    void *output, int32_t *dims, const uint32_t dims_size) {
  MACE_ASSERT1(output != NULL || dims != NULL || dims_size == 0,
               "Invalid param");
  MACE_ASSERT1(output_idx_ < kMaxOutputNum, "Not enough mem.");
  outputs_[output_idx_] = output;
  output_dims_[output_idx_] = dims;
  output_dim_sizes_[output_idx_] = dims_size;
  ++output_idx_;
  return *this;
}

uint32_t SubstituteOp::GetInputSize() {
  return input_idx_;
}

const void *SubstituteOp::DoGetInputData(uint32_t idx) {
  MACE_ASSERT1(idx < input_idx_, "idx is not valid");
  return inputs_[idx];
}

uint32_t SubstituteOp::GetInputShapeDimSize(uint32_t idx) {
  MACE_ASSERT1(idx < input_idx_, "idx is not valid");
  return input_dim_sizes_[idx];
}

const int32_t *SubstituteOp::GetInputShapeDims(uint32_t idx) {
  MACE_ASSERT1(idx < input_idx_, "idx is not valid");
  return input_dims_[idx];
}

uint32_t SubstituteOp::GetOutputSize() {
  return output_idx_;
}

void *SubstituteOp::DoGetOutputData(uint32_t idx) {
  MACE_ASSERT1(idx < output_idx_, "idx is not valid");
  return outputs_[idx];
}

uint32_t SubstituteOp::GetOutputShapeDimSize(uint32_t idx) {
  MACE_ASSERT1(idx < output_idx_, "idx is not valid");
  return output_dim_sizes_[idx];
}

const int32_t *SubstituteOp::GetOutputShapeDims(uint32_t idx) {
  MACE_ASSERT1(idx < output_idx_, "idx is not valid");
  return output_dims_[idx];
}

MaceStatus SubstituteOp::ResizeOutputShape(uint32_t idx,
                                           uint32_t input_dim_size,
                                           const int32_t *input_dims) {
  MACE_ASSERT1(idx < output_idx_, "idx is not valid");
  MACE_ASSERT1(input_dim_size <= output_dim_sizes_[idx],
               "Can not support dynamic dim size");
  if (output_dims_[idx] != NULL && input_dim_size > 0) {
    base::memcpy(output_dims_[idx], input_dims,
                 sizeof(int32_t) * input_dim_size);
  }
  output_dim_sizes_[idx] = input_dim_size;

  return MACE_SUCCESS;
}

MaceStatus SubstituteOp::ReuseInputBufferForOutput(uint32_t output_idx,
                                                   uint32_t input_idx) {
  MACE_UNUSED(output_idx);
  MACE_UNUSED(input_idx);
  return MACE_SUCCESS;
}

}  // namespace framework
}  // namespace micro
