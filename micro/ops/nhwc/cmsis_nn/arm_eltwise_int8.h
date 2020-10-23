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

#ifndef MICRO_OPS_NHWC_CMSIS_NN_ARM_ELTWISE_INT8_H_
#define MICRO_OPS_NHWC_CMSIS_NN_ARM_ELTWISE_INT8_H_

#include "micro/base/logger.h"
#include "micro/base/logging.h"
#include "micro/base/types.h"
#include "micro/base/utils.h"
#include "micro/framework/op_context.h"
#include "micro/framework/operator.h"
#include "micro/framework/scratch_buffer.h"
#include "micro/model/const_tensor.h"
#include "micro/model/net_def.h"

namespace micro {
namespace ops {

class ArmEltwiseInt8Op : public framework::Operator {
 public:
  MaceStatus OnInit();

  MaceStatus Run();

 private:
  const int8_t *input0_;
  const int32_t *input0_dims_;
  uint32_t input0_dim_size_;

  const int8_t *input1_;
  const int32_t *input1_dims_;
  uint32_t input1_dim_size_;

  int8_t *output_;

  eltwise::Type type_;
  const float *coeff_;
  uint32_t coeff_size_;
  int32_t scalar_input_index_;
  bool nchw_;

  MACE_OP_INPUT_TAGS(INPUT0, INPUT1);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace micro


#endif  // MICRO_OPS_NHWC_CMSIS_NN_ARM_ELTWISE_INT8_H_
