//
// Copyright (c) 2017 XiaoMi All rights reserved.
//


#ifndef MACE_CORE_RUNTIME_CPU_CPU_RUNTIME_H_
#define MACE_CORE_RUNTIME_CPU_CPU_RUNTIME_H_

#include <vector>

#include "mace/public/mace.h"
#include "mace/public/mace_runtime.h"

namespace mace {

MaceStatus GetCPUBigLittleCoreIDs(std::vector<int> *big_core_ids,
                                  std::vector<int> *little_core_ids);

void SetOpenMPThreadsAndAffinityCPUs(int omp_num_threads,
                                     const std::vector<int> &cpu_ids);

MaceStatus SetOpenMPThreadsAndAffinityPolicy(int omp_num_threads_hint,
                                             CPUAffinityPolicy policy);

}  // namespace mace

#endif  // MACE_CORE_RUNTIME_CPU_CPU_RUNTIME_H_
