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

namespace {
template<DeviceType D, typename T>
void BMSplitHelper(int iters,
                   const std::vector<index_t> &input_shape,
                   const index_t num_outputs) {
  mace::testing::StopTiming();

  // Construct graph
  OpsTestNet net;

  const index_t input_size = std::accumulate(input_shape.begin(),
                                             input_shape.end(),
                                             1,
                                             std::multiplies<index_t>());
  std::vector<float> input_data(input_size);
  GenerateRandomRealTypeData(input_shape, &input_data);
  net.AddInputFromArray<D, float>("Input", input_shape, input_data);

  auto builder = OpDefBuilder("Split", "SplitTest");
  builder.Input("Input");
  for (int i = 0; i < num_outputs; ++i) {
    builder = builder.Output(MakeString("Output", i));
  }
  builder
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .AddIntArg("has_data_format", 1)
      .Finalize(net.NewOperatorDef());

  // Warm-up
  for (int i = 0; i < 2; ++i) {
    net.RunOp(D);
    net.Sync();
  }

  mace::testing::StartTiming();
  while (iters--) {
    net.RunOp(D);
    net.Sync();
  }
}
}  // namespace

#define MACE_BM_SPLIT_MACRO(N, H, W, C, NO, TYPE, DEVICE)                    \
  static void                                                                \
      MACE_BM_SPLIT_##N##_##H##_##W##_##C##_##NO##_##TYPE##_##DEVICE(        \
          int iters) {                                                       \
        const int64_t tot = static_cast<int64_t>(iters) * N * H * W * C;     \
        mace::testing::MacsProcessed(tot);                                   \
        mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                  \
        BMSplitHelper<DEVICE, TYPE>(iters, {N, H, W, C}, NO);                \
      }                                                                      \
      MACE_BENCHMARK(                                                        \
          MACE_BM_SPLIT_##N##_##H##_##W##_##C##_##NO##_##TYPE##_##DEVICE)

#ifdef MACE_ENABLE_OPENCL
#define MACE_BM_SPLIT(N, H, W, C, NO)                 \
  MACE_BM_SPLIT_MACRO(N, H, W, C, NO, float, CPU);    \
  MACE_BM_SPLIT_MACRO(N, H, W, C, NO, float, GPU);    \
  MACE_BM_SPLIT_MACRO(N, H, W, C, NO, half, GPU)
#else
#define MACE_BM_SPLIT(N, H, W, C, NO)                 \
  MACE_BM_SPLIT_MACRO(N, H, W, C, NO, float, CPU)
#endif

MACE_BM_SPLIT(1, 32, 32, 32, 2);
MACE_BM_SPLIT(1, 32, 32, 128, 2);
MACE_BM_SPLIT(1, 32, 32, 256, 2);
MACE_BM_SPLIT(1, 128, 128, 32, 2);
MACE_BM_SPLIT(1, 128, 128, 128, 2);

}  // namespace test
}  // namespace ops
}  // namespace mace
