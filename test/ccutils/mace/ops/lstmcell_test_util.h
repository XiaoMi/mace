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

#ifndef MACE_OPS_LSTMCELL_TEST_UTIL_H_
#define MACE_OPS_LSTMCELL_TEST_UTIL_H_

#include <string>

#include "mace/ops/eltwise.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

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
      .AddIntArg("type", static_cast<int>(ops::EltwiseType::PROD))
      .Output("RememberMul")
      .Finalize(net->AddNewOperatorDef());

  OpDefBuilder("Eltwise", "ForgetAdd")
      .Input("SplitOutput2")
      .AddFloatArg("scalar_input", forget_add_name)
      .AddIntArg("T", DataTypeToEnum<T>::v())
      .AddIntArg("type", static_cast<int>(ops::EltwiseType::SUM))
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
      .AddIntArg("type", static_cast<int>(ops::EltwiseType::PROD))
      .Output("ForgetMulPreCell")
      .Finalize(net->AddNewOperatorDef());

  OpDefBuilder("Eltwise", "Cell")
      .Input("RememberMul")
      .Input("ForgetMulPreCell")
      .AddIntArg("T", DataTypeToEnum<T>::v())
      .AddIntArg("type", static_cast<int>(ops::EltwiseType::SUM))
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
      .AddIntArg("type", static_cast<int>(ops::EltwiseType::PROD))
      .Output(output_name)
      .Finalize(net->AddNewOperatorDef());
}

}  // namespace test
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_LSTMCELL_TEST_UTIL_H_
