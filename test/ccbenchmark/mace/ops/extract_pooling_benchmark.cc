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

#include "mace/benchmark_utils/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

template <DeviceType D, typename T>
static void ExtractPooling(int iters,
                           int batch,
                           int chunk,
                           int dim,
                           int input_period,
                           int modulus) {
  mace::testing::StopTiming();

  OpsTestNet net;

  size_t num_input_indexes = static_cast<size_t>(chunk / input_period);
  std::vector<int> input_indexes(num_input_indexes, 0);

  for (size_t i = 0; i < num_input_indexes; ++i) {
    input_indexes[i] = static_cast<int>(i * input_period);
  }

  size_t num_output_indexes = static_cast<size_t>(chunk / modulus);
  std::vector<int> output_indexes(num_output_indexes, 0);
  std::vector<int> forward_indexes(num_output_indexes * 2, 0);
  std::vector<float> counts(num_output_indexes, 0.f);
  for (size_t i = 0; i < num_output_indexes; ++i) {
    output_indexes[i] = static_cast<int>(i * modulus);
    forward_indexes[2 * i] = 0;
    forward_indexes[2 * i + 1] = static_cast<int>(num_input_indexes - 1);
    counts[i] = static_cast<float>(num_input_indexes);
  }

  // Add input data
  net.AddRandomInput<D, T>("Input", {batch, chunk, dim});

  OpDefBuilder("ExtractPooling", "ExtractPoolingTest")
      .Input("Input")
      .AddIntArg("modulus", modulus)
      .AddIntArg("include_variance", 1)
      .AddIntArg("num_log_counts", 1)
      .AddIntsArg("input_indexes", input_indexes)
      .AddIntsArg("output_indexes", output_indexes)
      .AddIntsArg("forward_indexes", forward_indexes)
      .AddFloatsArg("counts", counts)
      .AddIntsArg("input_time_range", {0, chunk - 1})
      .AddIntsArg("output_time_range", {0, chunk - 1})
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Warm-up
  for (int i = 0; i < 5; ++i) {
    net.RunOp(D);
  }
  net.Sync();

  mace::testing::StartTiming();
  while (iters--) {
    net.Run();
  }
  net.Sync();
}

#define MACE_BM_EXTRACTPOOLING_MACRO(N, C, D, INP, M, TYPE, DEVICE)            \
  static void MACE_BM_EXTRACTPOOLING_##N##_##C##_##D##_##INP##_##M##_##TYPE##\
_##DEVICE(                                                                     \
      int iters) {                                                             \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * D;               \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                        \
    ExtractPooling<DEVICE, TYPE>(iters, N, C, D, INP, M);                      \
  }                                                                            \
  MACE_BENCHMARK(MACE_BM_EXTRACTPOOLING_##N##_##C##_##D##_##INP##_##M##_##TYPE\
##_##DEVICE)

#define MACE_BM_EXTRACTPOOLING(N, C, D, INP, M)                 \
  MACE_BM_EXTRACTPOOLING_MACRO(N, C, D, INP, M, float, CPU);

MACE_BM_EXTRACTPOOLING(8, 40, 512, 2, 4);
MACE_BM_EXTRACTPOOLING(16, 80, 100, 3, 9);
MACE_BM_EXTRACTPOOLING(32, 60, 200, 6, 18);


}  // namespace test
}  // namespace ops
}  // namespace mace
