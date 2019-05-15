// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#ifndef MACE_OPS_COMMON_ELTWISE_TYPE_H_
#define MACE_OPS_COMMON_ELTWISE_TYPE_H_

namespace mace {
namespace ops {

enum EltwiseType {
  SUM = 0,
  SUB = 1,
  PROD = 2,
  DIV = 3,
  MIN = 4,
  MAX = 5,
  NEG = 6,
  ABS = 7,
  SQR_DIFF = 8,
  POW = 9,
  EQUAL = 10,
  FLOOR_DIV = 11,
  CLIP = 12,
  NONE = 13,
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_COMMON_ELTWISE_TYPE_H_
