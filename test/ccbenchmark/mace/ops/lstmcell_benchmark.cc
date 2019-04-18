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
void LSTMCell(int iters, int batch, int input_size, int hidden_units) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, float>("Input", {batch, input_size});
  net.AddRandomInput<D, float>("PreOutput", {batch, hidden_units}, true);
  net.AddRandomInput<D, float>("Weight", {input_size + hidden_units,
                                          4 * hidden_units}, true);
  net.AddRandomInput<D, float>("Bias", {4 * hidden_units}, true);
  net.AddRandomInput<D, float>("PreCell", {batch, hidden_units}, true);

  const float &forget_add = 0.0f;

  if (D == DeviceType::CPU) {
    net.CopyData<DeviceType::CPU, float>("Input", "InputCPU");
    net.CopyData<DeviceType::CPU, float>("PreOutput", "PreOutputCPU");
    net.CopyData<DeviceType::CPU, float>("Weight", "WeightCPU");
    net.CopyData<DeviceType::CPU, float>("Bias", "BiasCPU");
    net.CopyData<DeviceType::CPU, float>("PreCell", "PreCellCPU");

    LSTMCellCPU<float>(&net, "InputCPU", "PreOutputCPU", "WeightCPU", "BiasCPU",
                       "PreCellCPU", forget_add, "CellCPU", "OutputCPU");
  } else if (D == DeviceType::GPU) {
    OpDefBuilder("LSTMCell", "LSTMCellTest")
        .Input("Input")
        .Input("PreOutput")
        .Input("Weight")
        .Input("Bias")
        .Input("PreCell")
        .AddFloatArg("scalar_input", forget_add)
        .Output("Cell")
        .Output("Output")
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

#define MACE_BM_LSTMCELL_MACRO(N, INPUT_SIZE, HIDDEN_UNITS, TYPE, DEVICE)      \
  static void                                                                  \
      MACE_BM_LSTMCELL_##N##_##INPUT_SIZE##_##HIDDEN_UNITS##_##TYPE##_##DEVICE(\
        int iters) {                                                           \
    const int64_t macs =                                                       \
        static_cast<int64_t>(                                                  \
            iters) * N * (INPUT_SIZE + HIDDEN_UNITS) * 4 * HIDDEN_UNITS;       \
    const int64_t tot = static_cast<int64_t>(iters) * N * INPUT_SIZE;          \
    mace::testing::MacsProcessed(macs);                                        \
    mace::testing::BytesProcessed(tot * (sizeof(TYPE)));                       \
    LSTMCell<DEVICE, TYPE>(iters, N, INPUT_SIZE, HIDDEN_UNITS);                \
  }                                                                            \
  MACE_BENCHMARK(                                                              \
      MACE_BM_LSTMCELL_##N##_##INPUT_SIZE##_##HIDDEN_UNITS##_##TYPE##_##DEVICE)

#ifdef MACE_ENABLE_OPENCL
#define MACE_BM_LSTMCELL(N, INPUT_SIZE, HIDDEN_UNITS)                 \
  MACE_BM_LSTMCELL_MACRO(N, INPUT_SIZE, HIDDEN_UNITS, float, CPU);    \
  MACE_BM_LSTMCELL_MACRO(N, INPUT_SIZE, HIDDEN_UNITS, float, GPU);    \
  MACE_BM_LSTMCELL_MACRO(N, INPUT_SIZE, HIDDEN_UNITS, half, GPU)
#else
#define MACE_BM_LSTMCELL(N, INPUT_SIZE, HIDDEN_UNITS)                 \
  MACE_BM_LSTMCELL_MACRO(N, INPUT_SIZE, HIDDEN_UNITS, float, CPU)
#endif

MACE_BM_LSTMCELL(1, 64, 256);
MACE_BM_LSTMCELL(30, 64, 256);
MACE_BM_LSTMCELL(50, 64, 256);
MACE_BM_LSTMCELL(80, 64, 256);
}  // namespace test
}  // namespace ops
}  // namespace mace
