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
void BMSpliceHelper(int iters,
                    const std::vector<index_t> &input_shape,
                    const int left_context,
                    const int right_context,
                    const int const_component_dim) {
  mace::testing::StopTiming();

  // Construct graph
  OpsTestNet net;

  const index_t num_splice = left_context + right_context + 1;
  std::vector<int> contexts(num_splice);
  for (int i = 0; i < num_splice; ++i) {
    contexts[i] = left_context + i;
  }
  const index_t input_size = std::accumulate(input_shape.begin(),
                                             input_shape.end(),
                                             1,
                                             std::multiplies<index_t>());
  std::vector<float> input_data(input_size);
  GenerateRandomRealTypeData(input_shape, &input_data);
  net.AddInputFromArray<D, float>("Input", input_shape, input_data);

  OpDefBuilder("Splice", "SpliceBM")
      .Input("Input")
      .Output("Output")
      .AddIntsArg("context", contexts)
      .AddIntArg("const_component_dim", const_component_dim)
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());

  // Warm-up
  for (int i = 0; i < 5; ++i) {
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

#define MACE_BM_SPLICE_MACRO(N, H, W, L, R, C, TYPE, DEVICE)  \
  static void                                                                \
      MACE_BM_SPLICE_##N##_##H##_##W##_##L##_##R##_##C##_##TYPE##_##DEVICE(  \
          int iters) {                                                       \
        const int64_t tot = static_cast<int64_t>(iters) * N * H * W;         \
        mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                  \
        BMSpliceHelper<DEVICE, TYPE>(iters, {N, H, W}, L, R, C);             \
      }                                                                      \
      MACE_BENCHMARK(                                                        \
          MACE_BM_SPLICE_##N##_##H##_##W##_##L##_##R##_##C##_##TYPE##_##DEVICE)

#define MACE_BM_SPLICE(N, H, W, L, R, C)                 \
  MACE_BM_SPLICE_MACRO(N, H, W, L, R, C, float, CPU);

MACE_BM_SPLICE(1, 32, 32, 5, 5, 10);
MACE_BM_SPLICE(1, 32, 32, 7, 7, 5);
MACE_BM_SPLICE(1, 32, 32, 3, 3, 20);
MACE_BM_SPLICE(1, 128, 128, 9, 9, 100);
MACE_BM_SPLICE(1, 128, 128, 7, 7, 100);

}  // namespace test
}  // namespace ops
}  // namespace mace
