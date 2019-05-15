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
#include "mace/ops/lstmcell_test_util.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template <DeviceType D, typename T>
void LSTMNonlinear(int iters,
                   int batch,
                   int input_dim) {
  mace::testing::StopTiming();

  OpsTestNet net;

  int cell_dim = input_dim / 5;

  // Add input data
  net.AddRandomInput<D, float>("Input", {batch, input_dim});
  net.AddRandomInput<D, float>("Params",
                               {3, cell_dim},
                               true);
  if (D == DeviceType::CPU) {
    OpDefBuilder("LSTMNonlinear", "LSTMNonlinearTest")
        .Input("Input")
        .Input("Params")
        .Output("Output")
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());
  }  else {
    MACE_NOT_IMPLEMENTED;
  }

  // Warm-up
  for (int i = 0; i < 5; ++i) {
    net.RunOp(D);
  }
  net.Sync();

  mace::testing::StartTiming();
  while (iters--) {
    net.RunOp(D);
  }
  net.Sync();
}
}  // namespace

#define MACE_BM_LSTM_NONLIN_MACRO(N, IN_DIM, TYPE, DEVICE)          \
  static void                                                       \
      MACE_BM_LSTM_NONLIN_##N##_##IN_DIM##_##TYPE##_##DEVICE(\
        int iters) {                                                \
    const int64_t tot =                                             \
      static_cast<int64_t>(iters) * (N * IN_DIM + 3 * (IN_DIM / 5));\
    mace::testing::BytesProcessed(tot * (sizeof(TYPE)));            \
    LSTMNonlinear<DEVICE, TYPE>(iters, N, IN_DIM);                  \
  }                                                                 \
  MACE_BENCHMARK(                                                   \
      MACE_BM_LSTM_NONLIN_##N##_##IN_DIM##_##TYPE##_##DEVICE)

#define MACE_BM_LSTM_NONLIN(N, IN_DIM)                 \
  MACE_BM_LSTM_NONLIN_MACRO(N, IN_DIM, float, CPU);

MACE_BM_LSTM_NONLIN(50, 200);
MACE_BM_LSTM_NONLIN(50, 920);
MACE_BM_LSTM_NONLIN(80, 640);

}  // namespace test
}  // namespace ops
}  // namespace mace
