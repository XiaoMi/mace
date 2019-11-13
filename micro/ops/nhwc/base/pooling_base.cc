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

#include "micro/ops/nhwc/base/pooling_base.h"

#include "micro/base/logging.h"
#include "micro/include/utils/macros.h"
#include "micro/ops/nhwc/base/filter_op_base.h"

namespace micro {
namespace ops {

MaceStatus PoolingBase::OnInit() {
  MACE_ASSERT1(static_cast<DataFormat>(
                   GetArgByName("data_format",
                                static_cast<int32_t>(NHWC)))
                   != NCHW, "Only support NHWC");
  input_ = GetInputData<mifloat>(INPUT);
  input_dims_ = GetInputShapeDims(INPUT);
  input_dim_size_ = GetInputShapeDimSize(INPUT);

  output_ = GetOutputData<mifloat>(OUTPUT);
  output_dims_ = GetOutputShapeDims(OUTPUT);
  output_dim_size_ = GetOutputShapeDimSize(OUTPUT);

  kernel_ = GetRepeatArgByName<int32_t>("kernels");
  MACE_ASSERT(kernel_ != NULL);
  int32_t pooling_type =
      GetArgByName("pooling_type", static_cast<int32_t>(AVG));
  pooling_type_ = static_cast<PoolingType>(pooling_type);
  int32_t round_type = GetArgByName("round_mode", static_cast<int32_t>(FLOOR));
  round_type_ = static_cast<RoundType>(round_type);

  filter_dims_[0] = filter_dims_[3] = input_dims_[3];
  filter_dims_[1] = kernel_[0];
  filter_dims_[2] = kernel_[1];

  return FilterOpBase::OnInitBase();
}

MaceStatus PoolingBase::Run() {
  int32_t output_dims[4] = {0};
  InitPaddingAndOutputSize(input_dims_, filter_dims_, round_type_, output_dims);
  ResizeOutputShape(OUTPUT, 4, output_dims);

  int32_t pad_hw[2] = {padding_sizes_[0] / 2, padding_sizes_[1] / 2};
  if (pooling_type_ == MAX) {
    MaxPooling(input_, kernel_, strides_, dilations_, pad_hw);
  } else if (pooling_type_ == AVG) {
    AvgPooling(input_, kernel_, strides_, dilations_, pad_hw);
  } else {
    MACE_NOT_IMPLEMENTED;
  }
  return MACE_SUCCESS;
}

void PoolingBase::MaxPooling(const mifloat *input,
                             const int32_t *filter_hw,
                             const int32_t *stride_hw,
                             const int32_t *dilation_hw,
                             const int32_t *pad_hw) {
  MACE_UNUSED(input);
  MACE_UNUSED(filter_hw);
  MACE_UNUSED(stride_hw);
  MACE_UNUSED(dilation_hw);
  MACE_UNUSED(pad_hw);
  MACE_NOT_IMPLEMENTED;
}

void PoolingBase::AvgPooling(const mifloat *input,
                             const int32_t *filter_hw,
                             const int32_t *stride_hw,
                             const int32_t *dilation_hw,
                             const int32_t *pad_hw) {
  MACE_UNUSED(input);
  MACE_UNUSED(filter_hw);
  MACE_UNUSED(stride_hw);
  MACE_UNUSED(dilation_hw);
  MACE_UNUSED(pad_hw);
  MACE_NOT_IMPLEMENTED;
}

}  // namespace ops
}  // namespace micro
