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

#include "mace/ops/eltwise.h"
#include "mace/ops/lstmcell_test_util.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class LSTMCellTest : public OpsTestBase {};

namespace {
template <DeviceType D, typename T>
void TestLSTMCell(const uint32_t &batch,
                  const uint32_t &input_size,
                  const uint32_t &hidden_units,
                  const float &forget_add) {
  // Construct graph
  OpsTestNet net;

  net.AddRandomInput<D, float>("Input", {batch, input_size});
  net.AddRandomInput<D, float>("PreOutput", {batch, hidden_units}, true);
  net.AddRandomInput<D, float>("Weight", {input_size + hidden_units,
                                          4 * hidden_units}, true);
  net.AddRandomInput<D, float>("Bias", {4 * hidden_units}, true);
  net.AddRandomInput<D, float>("PreCell", {batch, hidden_units}, true);

  net.CopyData<DeviceType::CPU, float>("Input", "InputCPU");
  net.CopyData<DeviceType::CPU, float>("PreOutput", "PreOutputCPU");
  net.CopyData<DeviceType::CPU, float>("Weight", "WeightCPU");
  net.CopyData<DeviceType::CPU, float>("Bias", "BiasCPU");
  net.CopyData<DeviceType::CPU, float>("PreCell", "PreCellCPU");

  // Run on CPU
  LSTMCellCPU<float>(&net, "InputCPU", "PreOutputCPU", "WeightCPU", "BiasCPU",
                     "PreCellCPU", forget_add, "CellCPU", "OutputCPU");
  // Run
  net.RunOp(DeviceType::CPU);

  // Run on GPU
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

  // Run
  net.RunOp(D);

  Tensor expected_cell, expected_output;
  expected_cell.Copy(*net.GetOutput("CellCPU"));
  expected_output.Copy(*net.GetOutput("OutputCPU"));

  if (DataTypeToEnum<T>::value == DT_HALF) {
    ExpectTensorNear<float>(expected_cell, *net.GetOutput("Cell"), 1e-3);
    ExpectTensorNear<float>(expected_output, *net.GetOutput("Output"), 1e-3);
  } else {
    ExpectTensorNear<float>(expected_cell, *net.GetOutput("Cell"), 1e-5);
    ExpectTensorNear<float>(expected_output, *net.GetOutput("Output"), 1e-5);
  }
}
}  // namespace

TEST_F(LSTMCellTest, OPENCLRandomHalf) {
  TestLSTMCell<GPU, half>(1, 3, 8, 0.0f);
  TestLSTMCell<GPU, half>(2, 16, 24, 0.0f);
  TestLSTMCell<GPU, half>(2, 200, 280, 0.5f);
  TestLSTMCell<GPU, half>(20, 320, 512, 0.5f);
}

TEST_F(LSTMCellTest, OPENCLRandomFloat) {
  TestLSTMCell<GPU, float>(1, 3, 8, 0.0f);
  TestLSTMCell<GPU, float>(2, 16, 24, 0.0f);
  TestLSTMCell<GPU, float>(2, 200, 280, 0.5f);
  TestLSTMCell<GPU, float>(20, 320, 512, 0.5f);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
