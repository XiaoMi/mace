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

#ifndef MICRO_OPS_UTILS_CRUMB_UTILS_H_
#define MICRO_OPS_UTILS_CRUMB_UTILS_H_

#include "micro/base/types.h"
#include "micro/include/public/micro.h"

namespace micro {
namespace ops {
namespace crumb {

MaceStatus ComputeBias(const mifloat *input, const int32_t *input_dims,
                       const uint32_t input_dim_size,
                       const mifloat *bias, const int32_t channel,
                       mifloat *output);

}  // crumb
}  // namespace ops
}  // namespace micro


#endif  // MICRO_OPS_UTILS_CRUMB_UTILS_H_
