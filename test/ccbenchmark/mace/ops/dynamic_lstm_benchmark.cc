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

#include "mace/utils/statistics.h"
#include "mace/benchmark_utils/test_benchmark.h"
#include "mace/ops/lstmcell_test_util.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template <DeviceType D, typename T>
void DynamicLSTM(int iters,
                 int chunk,
                 int input_dim,
                 int output_dim,
                 int cell_dim,
                 int prev_out_dim,
                 int delay) {
  mace::testing::StopTiming();

  OpsTestNet net;
  MACE_CHECK(prev_out_dim <= output_dim);
  const int weights_a_rows = 4 * cell_dim;
  const int weights_a_cols = input_dim + prev_out_dim;
  const int bias_a_rows = weights_a_rows;

  const int weights_b_rows = output_dim;
  const int weights_b_cols = cell_dim;
  const int bias_b_rows = weights_b_rows;

  // Add input data
  net.AddRandomInput<D, float>("Input", {chunk, input_dim});
  net.AddRandomInput<D, float>("Weight_A",
                               {weights_a_rows, weights_a_cols},
                               true);
  net.AddRandomInput<D, float>("Params",
                               {3, cell_dim},
                               true);
  net.AddRandomInput<D, float>("Weight_B",
                               {weights_b_rows, weights_b_cols},
                               true);
  net.AddRandomInput<D, float>("Bias_A", {bias_a_rows}, true);
  net.AddRandomInput<D, float>("Bias_B", {bias_b_rows}, true);

  if (D == DeviceType::CPU) {
    OpDefBuilder("DynamicLSTM", "DynamicLSTMTest")
        .Input("Input")
        .Input("Weight_A")
        .Input("Params")
        .Input("Weight_B")
        .Input("Bias_A")
        .Input("Bias_B")
        .Output("Output")
        .AddIntArg("prev_out_delay", -delay)
        .AddIntArg("prev_cell_delay", -delay)
        .AddIntArg("prev_out_dim", prev_out_dim)
        .AddIntArg("prev_cell_dim", cell_dim)
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

#define MACE_BM_DYNAMIC_LSTM_MACRO(                                           \
    N, ID, OD, CD, POD, DELAY, TYPE, DEVICE)                                  \
  static void                                                                 \
      MACE_BM_DYNAMIC_LSTM_##N##_##ID##_##OD##_##CD##_##POD##_##DELAY##_##TYPE\
        ##_##DEVICE(                                                          \
        int iters) {                                                          \
    int64_t wa_size = 4 * CD * (ID + POD);                                    \
    int64_t wb_size = OD * CD;                                                \
    int64_t prev_size = DELAY * (POD + CD);                                   \
    int64_t in_out_size = N * (ID + OD);                                      \
    int64_t bias_size = 4 * CD + OD;                                          \
    const int64_t macs = static_cast<int64_t>(iters) *                        \
        mace::benchmark::StatMACs("DynamicLSTM", {4 * CD, ID + POD}, {N, OD});\
    const int64_t tot = static_cast<int64_t>(iters) * (in_out_size + prev_size\
      + wa_size + wb_size + bias_size);                                       \
    mace::testing::MacsProcessed(macs);                                       \
    mace::testing::BytesProcessed(tot * (sizeof(TYPE)));                      \
    DynamicLSTM<DEVICE, TYPE>(iters, N, ID, OD, CD, POD, DELAY);              \
  }                                                                           \
  MACE_BENCHMARK(                                                             \
      MACE_BM_DYNAMIC_LSTM_##N##_##ID##_##OD##_##CD##_##POD##_##DELAY         \
        ##_##TYPE##_##DEVICE)

#define MACE_BM_DYNAMIC_LSTM(N, ID, OD, CD, POD, DELAY)                       \
  MACE_BM_DYNAMIC_LSTM_MACRO(N, ID, OD, CD, POD, DELAY, float, CPU);

MACE_BM_DYNAMIC_LSTM(50, 184, 128, 184, 64, 3);
MACE_BM_DYNAMIC_LSTM(50, 64, 256, 64, 128,  3);
MACE_BM_DYNAMIC_LSTM(80, 64, 256, 128, 64, 3);

}  // namespace test
}  // namespace ops
}  // namespace mace
