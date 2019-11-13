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

#include "micro/ops/nhwc/batch_norm.h"

#include "micro/base/logging.h"
#include "micro/base/utils.h"
#include "micro/framework/scratch_buffer.h"

namespace micro {
namespace ops {

MaceStatus BatchNormOp::OnInit() {
  input_ = GetInputData<mifloat>(INPUT);
  input_dims_ = GetInputShapeDims(INPUT);
  input_dim_size_ = GetInputShapeDimSize(INPUT);

  scale_ = GetInputData<mifloat>(SCALE);
  scale_dims_ = GetInputShapeDims(SCALE);
  scale_dim_size_ = GetInputShapeDimSize(SCALE);

  offset_ = GetInputData<mifloat>(OFFSET);
  offset_dims_ = GetInputShapeDims(OFFSET);
  offset_dim_size_ = GetInputShapeDimSize(OFFSET);

  output_ = GetOutputData<mifloat>(OUTPUT);

  MACE_ASSERT(input_dim_size_ >= 1);
  MACE_ASSERT1(scale_dim_size_ == 1, "scale must be 1-dimensional. ");
  MACE_ASSERT1(offset_dim_size_ == 1, "offset must be 1-dimensional. ");

  epsilon_ = GetArgByName("epsilon", static_cast<float>(1e-4));

  MACE_RETURN_IF_ERROR(activation_.Init(this));

  MACE_RETURN_IF_ERROR(ResizeOutputShape(OUTPUT, input_dim_size_, input_dims_));

  return MACE_SUCCESS;
}

MaceStatus BatchNormOp::Run() {
  const mifloat *scale = scale_;
  const mifloat *offset = offset_;
  const uint32_t input_dim_end_idx = input_dim_size_ - 1;
  const int32_t channels = input_dims_[input_dim_end_idx];
  const int32_t batch =
      base::accumulate_multi(input_dims_, 0, input_dim_end_idx);
  if (GetInputSize() == 5) {
    const float *mean = GetInputData<float>(MEAN);
    const float *var = GetInputData<float>(VAR);

    MACE_ASSERT1(GetInputShapeDimSize(MEAN) == 1,
                 "mean must be 1-dimensional. ");
    MACE_ASSERT1(GetInputShapeDimSize(VAR) == 1, "var must be 1-dimensional. ");

    ScratchBuffer scratch_buffer(engine_config_);
    mifloat *new_scale = scratch_buffer.GetBuffer<mifloat>(channels);
    mifloat *new_offset = scratch_buffer.GetBuffer<mifloat>(channels);
    for (int32_t c = 0; c < channels; ++c) {
      new_scale[c] = scale_[c] / base::sqrt(var[c] + epsilon_);
      new_offset[c] = offset_[c] - mean[c] * new_scale[c];
    }
    scale = new_scale;
    offset = new_offset;
  }

  for (int32_t b = 0; b < batch; ++b) {
    const int32_t batch_base = b * channels;
    for (int32_t c = 0; c < channels; ++c) {
      output_[batch_base + c] =
          input_[batch_base + c] * scale[c] + offset[c];
    }  // c
  }  // b

  MACE_RETURN_IF_ERROR(activation_.Compute(output_, batch * channels, output_));

  return MACE_SUCCESS;
}

}  // namespace ops
}  // namespace micro
