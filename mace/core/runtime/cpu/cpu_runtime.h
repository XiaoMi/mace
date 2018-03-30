//
// Copyright (c) 2017 XiaoMi All rights reserved.
//


#ifndef MACE_CORE_RUNTIME_CPU_CPU_RUNTIME_H_
#define MACE_CORE_RUNTIME_CPU_CPU_RUNTIME_H_

#include "mace/public/mace_runtime.h"

namespace mace {

void SetOmpThreads(int omp_num_threads);

void SetThreadsAffinity(CPUPowerOption power_option);

}

#endif  // MACE_CORE_RUNTIME_CPU_CPU_RUNTIME_H_
