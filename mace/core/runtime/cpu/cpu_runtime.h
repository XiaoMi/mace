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

#ifndef MACE_CORE_RUNTIME_CPU_CPU_RUNTIME_H_
#define MACE_CORE_RUNTIME_CPU_CPU_RUNTIME_H_

#include <vector>

#include "mace/public/mace.h"
#include "mace/public/mace_runtime.h"

namespace mace {

MaceStatus GetCPUBigLittleCoreIDs(std::vector<int> *big_core_ids,
                                  std::vector<int> *little_core_ids);

MaceStatus SetOpenMPThreadsAndAffinityCPUs(int omp_num_threads,
                                           const std::vector<int> &cpu_ids);

MaceStatus SetOpenMPThreadsAndAffinityPolicy(int omp_num_threads_hint,
                                             CPUAffinityPolicy policy);

}  // namespace mace

#endif  // MACE_CORE_RUNTIME_CPU_CPU_RUNTIME_H_
