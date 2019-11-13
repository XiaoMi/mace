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

#ifndef MICRO_OPS_UTILS_ACTIVATION_H_
#define MICRO_OPS_UTILS_ACTIVATION_H_

#include "micro/base/types.h"
#include "micro/include/public/micro.h"

namespace micro {
namespace framework {
class Operator;
}  // namespace framework

namespace ops {

enum ActivationType {
  NOOP = 0,
  RELU = 1,
  RELUX = 2,
  PRELU = 3,
  TANH = 4,
  SIGMOID = 5,
  LEAKYRELU = 6,

  TYPE_COUNT,
};

class Activation {
 public:
  Activation();
  ~Activation() {}

  MaceStatus Init(const framework::Operator *op);
  MaceStatus Init(const char *type, const float limit,
                  const float leakyrelu_coefficient);
  MaceStatus Compute(const mifloat *input_ptr,
                     const int32_t size, mifloat *output_ptr);
  ActivationType GetActivationType();

 private:
  ActivationType StringToActivationType(const char *type);

 private:
  ActivationType type_;
  float limit_;
  float leakyrelu_coefficient_;
};

}  // namespace ops
}  // namespace micro


#endif  // MICRO_OPS_UTILS_ACTIVATION_H_
