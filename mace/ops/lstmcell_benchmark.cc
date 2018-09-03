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

#include "mace/core/operator.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template <DeviceType D, typename T>
void LSTMCell(int iters, int batch, int lstm_step) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  if (D == DeviceType::GPU) {
    net.AddRandomInput<D, T>("Input", {batch, lstm_step});
    net.AddRandomInput<D, T>("PreOutput", {batch, lstm_step});
    net.AddRandomInput<D, T>("Weight", {2 * lstm_step, 4 * lstm_step});
    net.AddRandomInput<D, T>("Bias", {4 * lstm_step});
    net.AddRandomInput<D, T>("PreCell", {batch, lstm_step});
  } else {
    MACE_NOT_IMPLEMENTED;
  }

  if (D == DeviceType::GPU) {
    BufferToImage<D, T>(&net, "Input", "InputImage",
                        kernels::BufferType::IN_OUT_CHANNEL);
    BufferToImage<D, T>(&net, "PreOutput", "PreOutputImage",
                        kernels::BufferType::IN_OUT_CHANNEL);
    BufferToImage<D, T>(&net, "Weight", "WeightImage",
                        kernels::BufferType::IN_OUT_CHANNEL);
    BufferToImage<D, T>(&net, "Bias", "BiasImage",
                        kernels::BufferType::ARGUMENT);
    BufferToImage<D, T>(&net, "PreCell", "PreCellImage",
                      kernels::BufferType::IN_OUT_CHANNEL);

    OpDefBuilder("LSTMCell", "LSTMCellTest")
        .Input("InputImage")
        .Input("PreOutputImage")
        .Input("WeightImage")
        .Input("BiasImage")
        .Input("PreCellImage")
        .AddFloatArg("forget_add", 0.0f)
        .Output("CellImage")
        .Output("OutputImage")
        .Finalize(net.NewOperatorDef());
  } else {
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

#define MACE_BM_LSTMCELL_MACRO(N, LSTM_STEP, TYPE, DEVICE)                \
  static void MACE_BM_LSTMCELL_##N##_##LSTM_STEP##_##TYPE##_##DEVICE(     \
      int iters) {                                                        \
    const int64_t macc =                                                  \
        static_cast<int64_t>(iters) * N * 2 * LSTM_STEP * 4 * LSTM_STEP;  \
    const int64_t tot = static_cast<int64_t>(iters) * N * LSTM_STEP;      \
    mace::testing::MaccProcessed(macc);                                   \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                   \
    LSTMCell<DEVICE, TYPE>(iters, N, LSTM_STEP);                          \
  }                                                                       \
  MACE_BENCHMARK(MACE_BM_LSTMCELL_##N##_##LSTM_STEP##_##TYPE##_##DEVICE)

#define MACE_BM_LSTMCELL(N, LSTM_STEP)                 \
  MACE_BM_LSTMCELL_MACRO(N, LSTM_STEP, float, GPU);    \
  MACE_BM_LSTMCELL_MACRO(N, LSTM_STEP, half, GPU);

MACE_BM_LSTMCELL(1, 200);
MACE_BM_LSTMCELL(20, 200);
MACE_BM_LSTMCELL(20, 320);
MACE_BM_LSTMCELL(32, 400);
MACE_BM_LSTMCELL(32, 640);
}  // namespace test
}  // namespace ops
}  // namespace mace
