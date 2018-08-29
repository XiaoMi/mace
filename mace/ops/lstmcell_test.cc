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
#include "mace/kernels/eltwise.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class LSTMCellTest : public OpsTestBase {};

namespace {

template <typename T>
void LSTMCellCPU(OpsTestNet *net,
                 const std::string &input_name,
                 const std::string &pre_output_name,
                 const std::string &weight_name,
                 const std::string &bias_name,
                 const std::string &pre_cell_name,
                 const float &forget_add_name,
                 const std::string &cell_name,
                 const std::string &output_name) {
  OpDefBuilder("Concat", "Concat")
      .Input(input_name)
      .Input(pre_output_name)
      .AddIntArg("axis", 1)
      .Output("ConcatOutput")
      .Finalize(net->AddNewOperatorDef());

  OpDefBuilder("MatMul", "MatMul")
      .Input("ConcatOutput")
      .Input(weight_name)
      .Output("MatMulOutput")
      .Finalize(net->AddNewOperatorDef());

  OpDefBuilder("BiasAdd", "BiasAdd")
      .Input("MatMulOutput")
      .Input(bias_name)
      .Output("BiasOutput")
      .Finalize(net->AddNewOperatorDef());

  OpDefBuilder("Split", "FCSplit")
      .Input("BiasOutput")
      .AddIntArg("axis", 1)
      .Output("SplitOutput0")
      .Output("SplitOutput1")
      .Output("SplitOutput2")
      .Output("SplitOutput3")
      .Finalize(net->AddNewOperatorDef());

  OpDefBuilder("Activation", "InputSigmoid")
      .Input("SplitOutput0")
      .AddStringArg("activation", "SIGMOID")
      .Output("InputSigmoid")
      .Finalize(net->AddNewOperatorDef());

  OpDefBuilder("Activation", "NewInputTanh")
      .Input("SplitOutput1")
      .AddStringArg("activation", "TANH")
      .Output("NewInputTanh")
      .Finalize(net->AddNewOperatorDef());

  OpDefBuilder("Eltwise", "RememberMul")
      .Input("InputSigmoid")
      .Input("NewInputTanh")
      .AddIntArg("T", DataTypeToEnum<T>::v())
      .AddIntArg("type", static_cast<int>(kernels::EltwiseType::PROD))
      .Output("RememberMul")
      .Finalize(net->AddNewOperatorDef());

  OpDefBuilder("Eltwise", "ForgetAdd")
      .Input("SplitOutput2")
      .AddFloatArg("value", forget_add_name)
      .AddIntArg("T", DataTypeToEnum<T>::v())
      .AddIntArg("type", static_cast<int>(kernels::EltwiseType::SUM))
      .Output("ForgetAdd")
      .Finalize(net->AddNewOperatorDef());

  OpDefBuilder("Activation", "ForgetSigmoid")
      .Input("ForgetAdd")
      .AddStringArg("activation", "SIGMOID")
      .Output("ForgetSigmoid")
      .Finalize(net->AddNewOperatorDef());

  OpDefBuilder("Eltwise", "ForgetMul")
      .Input("ForgetSigmoid")
      .Input(pre_cell_name)
      .AddIntArg("T", DataTypeToEnum<T>::v())
      .AddIntArg("type", static_cast<int>(kernels::EltwiseType::PROD))
      .Output("ForgetMulPreCell")
      .Finalize(net->AddNewOperatorDef());

  OpDefBuilder("Eltwise", "Cell")
      .Input("RememberMul")
      .Input("ForgetMulPreCell")
      .AddIntArg("T", DataTypeToEnum<T>::v())
      .AddIntArg("type", static_cast<int>(kernels::EltwiseType::SUM))
      .Output(cell_name)
      .Finalize(net->AddNewOperatorDef());

  OpDefBuilder("Activation", "CellTanh")
      .Input(cell_name)
      .AddStringArg("activation", "TANH")
      .Output("CellTanh")
      .Finalize(net->AddNewOperatorDef());

  OpDefBuilder("Activation", "OutputSigmoid")
      .Input("SplitOutput3")
      .AddStringArg("activation", "SIGMOID")
      .Output("OutputSigmoid")
      .Finalize(net->AddNewOperatorDef());

  OpDefBuilder("Eltwise", "FinalMul")
      .Input("OutputSigmoid")
      .Input("CellTanh")
      .AddIntArg("T", DataTypeToEnum<T>::v())
      .AddIntArg("type", static_cast<int>(kernels::EltwiseType::PROD))
      .Output(output_name)
      .Finalize(net->AddNewOperatorDef());
}

template <DeviceType D, typename T>
void TestLSTMCell(const uint32_t &batch,
                  const uint32_t &lstm_step,
                  const float &forget_add) {
  // Construct graph
  OpsTestNet net;

  net.AddRandomInput<D, float>("Input", {batch, lstm_step});
  net.AddRandomInput<D, float>("PreOutput", {batch, lstm_step});
  net.AddRandomInput<D, float>("Weight", {2 * lstm_step, 4 * lstm_step});
  net.AddRandomInput<D, float>("Bias", {4 * lstm_step});
  net.AddRandomInput<D, float>("PreCell", {batch, lstm_step});

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
      .AddFloatArg("forget_add", forget_add)
      .Output("CellImage")
      .Output("OutputImage")
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(D);

  ImageToBuffer<D, float>(&net, "OutputImage", "Output",
                      kernels::BufferType::IN_OUT_CHANNEL);
  ImageToBuffer<D, float>(&net, "CellImage", "Cell",
                      kernels::BufferType::IN_OUT_CHANNEL);


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
  TestLSTMCell<GPU, half>(1, 4, 0.0f);
  TestLSTMCell<GPU, half>(2, 16, 0.0f);
  TestLSTMCell<GPU, half>(2, 200, 0.5f);
  TestLSTMCell<GPU, half>(20, 320, 0.5f);
}

TEST_F(LSTMCellTest, OPENCLRandomFloat) {
  TestLSTMCell<GPU, float>(1, 4, 0.0f);
  TestLSTMCell<GPU, float>(2, 16, 0.0f);
  TestLSTMCell<GPU, float>(2, 200, 0.5f);
  TestLSTMCell<GPU, float>(20, 320, 0.5f);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
