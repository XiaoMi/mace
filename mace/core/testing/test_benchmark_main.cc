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

#include <iostream>

#include "gflags/gflags.h"
#include "mace/core/runtime/cpu/cpu_runtime.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/public/mace.h"
#include "mace/public/mace_runtime.h"
#include "mace/utils/logging.h"

DEFINE_string(filter, "all", "op benchmark regex filter, eg:.*CONV.*");
DEFINE_int32(gpu_perf_hint, 3, "0:DEFAULT/1:LOW/2:NORMAL/3:HIGH");
DEFINE_int32(gpu_priority_hint, 3, "0:DEFAULT/1:LOW/2:NORMAL/3:HIGH");
DEFINE_int32(omp_num_threads, -1, "num of openmp threads");
DEFINE_int32(cpu_affinity_policy, 1,
             "0:AFFINITY_NONE/1:AFFINITY_BIG_ONLY/2:AFFINITY_LITTLE_ONLY");

int main(int argc, char **argv) {
  std::string usage = "run ops benchmark\nusage: " + std::string(argv[0])
      + " [flags]";
  gflags::SetUsageMessage(usage);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // config runtime
  mace::MaceStatus status = mace::SetOpenMPThreadsAndAffinityPolicy(
      FLAGS_omp_num_threads,
      static_cast<mace::CPUAffinityPolicy>(FLAGS_cpu_affinity_policy));
  if (status != mace::MACE_SUCCESS) {
    LOG(WARNING) << "Set openmp or cpu affinity failed.";
  }
  status = SetGemmlowpThreadPolicy(
      FLAGS_omp_num_threads,
      static_cast<mace::CPUAffinityPolicy>(FLAGS_cpu_affinity_policy));
  if (status != mace::MACE_SUCCESS) {
    LOG(WARNING) << "Set gemmlowp threads or cpu affinity failed.";
  }
  mace::OpenCLRuntime::Configure(
      static_cast<mace::GPUPerfHint>(FLAGS_gpu_perf_hint),
      static_cast<mace::GPUPriorityHint>(FLAGS_gpu_priority_hint));

  mace::testing::Benchmark::Run(FLAGS_filter.c_str());
  return 0;
}
