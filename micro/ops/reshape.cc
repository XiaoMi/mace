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

#include "micro/ops/reshape.h"

#include "micro/base/logging.h"
#include "micro/base/utils.h"
#include "micro/framework/scratch_buffer.h"

namespace micro {
namespace ops {

namespace {

MaceStatus ValidShapeData(const int32_t *input_dims,
                          const uint32_t input_dim_size,
                          int32_t *shape_data,
                          const uint32_t shape_data_size) {
  MACE_ASSERT(
      input_dims != NULL && shape_data != NULL);
  int32_t unknown_idx = -1;
  int32_t product = 1;
  const int32_t input_size = base::GetShapeSize(input_dim_size, input_dims);

  for (uint32_t i = 0; i < shape_data_size; ++i) {
    if (shape_data[i] == -1) {
      MACE_ASSERT1(unknown_idx == -1, "Only one input size may be -1");
      unknown_idx = i;
      shape_data[i] = 1;
    } else {
      MACE_ASSERT2(shape_data[i] >= 0, "Shape must be non-negative: ",
                   shape_data[i]);
      if (shape_data[i] == 0) {
        MACE_ASSERT1(i < input_dim_size, "dims:0 out of input dims' range.");
        shape_data[i] = input_dims[i];
      }
      product *= shape_data[i];
    }
  }

  if (unknown_idx != -1) {
    MACE_ASSERT1(product != 0,
                 "Cannot infer shape if there is zero shape size.");
    const int32_t missing = input_size / product;
    MACE_ASSERT1(missing * product == input_size,
                 "Input size not match reshaped tensor size");
    shape_data[unknown_idx] = missing;
  }

  return MACE_SUCCESS;
}

}  // namespace

MaceStatus ReshapeOp::OnInit() {
  input_ = GetInputData<mifloat>(INPUT);
  input_dims_ = GetInputShapeDims(INPUT);
  input_dim_size_ = GetInputShapeDimSize(INPUT);

  shape_ = GetInputData<int32_t>(SHAPE);
  shape_dims_ = GetInputShapeDims(SHAPE);
  shape_dim_size_ = GetInputShapeDimSize(SHAPE);

  output_ = GetOutputData<mifloat>(OUTPUT);
  return MACE_SUCCESS;
}

MaceStatus ReshapeOp::Run() {
  const int32_t input_data_size =
      base::GetShapeSize(input_dim_size_, input_dims_);
  const int32_t shape_data_size =
      base::GetShapeSize(shape_dim_size_, shape_dims_);

  int32_t *shape_data =
      ScratchBuffer(engine_config_).GetBuffer<int32_t>(shape_data_size);
  base::memcpy(shape_data, shape_, shape_data_size * sizeof(int32_t));

  MACE_RETURN_IF_ERROR(ValidShapeData(input_dims_, input_dim_size_,
                                      shape_data, shape_data_size));

#ifndef NDEBUG
  const int32_t output_data_size = base::accumulate_multi(
      shape_data, 0, static_cast<uint32_t>(shape_data_size));
  if (input_data_size != output_data_size) {
    LOG(FATAL) << "input_data_size(" << input_data_size
               << ") != output_data_size(" << output_data_size
               << "), please check the model.";
  }
#endif

  // TODO(luxuhui): optimize this method by reusing buffer
  base::memcpy(output_, input_, input_data_size * sizeof(mifloat));
  return ResizeOutputShape(OUTPUT, shape_data_size, shape_data);
}

}  // namespace ops
}  // namespace micro
