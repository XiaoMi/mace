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

#ifndef MICRO_INCLUDE_UTILS_MACROS_H_
#define MICRO_INCLUDE_UTILS_MACROS_H_

#include "micro/include/public/micro.h"

namespace micro {

#ifndef MACE_EMPTY_VIRTUAL_DESTRUCTOR
#define MACE_EMPTY_VIRTUAL_DESTRUCTOR(CLASSNAME) \
 public:                                         \
  virtual ~CLASSNAME() {}
#endif  // MACE_EMPTY_VIRTUAL_DESTRUCTOR

#define MACE_UNUSED(var) (void)(var)

}  // namespace micro

#endif  // MICRO_INCLUDE_UTILS_MACROS_H_
