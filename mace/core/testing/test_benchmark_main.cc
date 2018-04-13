//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <iostream>

#include "gflags/gflags.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/public/mace.h"
#include "mace/public/mace_runtime.h"

DEFINE_string(filter, "all", "op benchmark regex filter, eg:.*CONV.*");
DEFINE_int32(gpu_perf_hint, 3, "0:DEFAULT/1:LOW/2:NORMAL/3:HIGH");
DEFINE_int32(gpu_priority_hint, 3, "0:DEFAULT/1:LOW/2:NORMAL/3:HIGH");
DEFINE_int32(omp_num_threads, -1, "num of openmp threads");
DEFINE_int32(cpu_affinity_policy, 1,
             "0:AFFINITY_DEFAULT/1:AFFINITY_BIG_ONLY/2:AFFINITY_LITTLE_ONLY");

int main(int argc, char **argv) {
  gflags::SetUsageMessage("some usage message");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // config runtime
  mace::SetOpenMPThreadPolicy(
      FLAGS_omp_num_threads,
      static_cast<mace::CPUAffinityPolicy >(FLAGS_cpu_affinity_policy));
  mace::SetGPUHints(
      static_cast<mace::GPUPerfHint>(FLAGS_gpu_perf_hint),
      static_cast<mace::GPUPriorityHint>(FLAGS_gpu_priority_hint));

  mace::testing::Benchmark::Run(FLAGS_filter.c_str());
  return 0;
}
