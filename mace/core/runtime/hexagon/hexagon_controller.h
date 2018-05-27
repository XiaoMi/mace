// Copyright 2018 Xiaomi, Inc.  All rights reserved.
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

#ifndef MACE_CORE_RUNTIME_HEXAGON_HEXAGON_CONTROLLER_H_
#define MACE_CORE_RUNTIME_HEXAGON_HEXAGON_CONTROLLER_H_

#include "third_party/nnlib/hexagon_nn.h"

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif  // __cplusplus

int hexagon_controller_InitHexagonWithMaxAttributes(int enable_dcvs,
                                                    int bus_usage);

int hexagon_controller_DeInitHexagon();

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // MACE_CORE_RUNTIME_HEXAGON_HEXAGON_CONTROLLER_H_

