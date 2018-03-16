//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_RUNTIME_HEXAGON_HEXAGON_CONTROLLER_H_
#define MACE_CORE_RUNTIME_HEXAGON_HEXAGON_CONTROLLER_H_

#include "mace/core/runtime/hexagon/hexagon_nn.h"

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

