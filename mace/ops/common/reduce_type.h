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

#ifndef MACE_OPS_COMMON_REDUCE_TYPE_H_
#define MACE_OPS_COMMON_REDUCE_TYPE_H_


namespace mace {
enum ReduceType {
  MEAN = 0,
  MIN = 1,
  MAX = 2,
  PROD = 3,
  SUM = 4,
//  SUM_SQR = 4,
//  SQR_MEAN = 5,
};
}  // namespace mace

#endif  // MACE_OPS_COMMON_REDUCE_TYPE_H_
