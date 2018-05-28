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
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class QuantizeTest : public OpsTestBase {};

TEST_F(QuantizeTest, TestQuantize) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<CPU, float>("Input", {1, 2, 3, 1},
                                    {-2, -1, 1, 2, 3, 4});
  net.AddInputFromArray<CPU, float>("InputMin", {1}, {-3});
  net.AddInputFromArray<CPU, float>("InputMax", {1}, {5});

  OpDefBuilder("Quantize", "QuantizeTest")
      .Input("Input")
      .Input("InputMin")
      .Input("InputMax")
      .Output("Output")
      .Output("OutputMin")
      .Output("OutputMax")
      .OutputType({DT_UINT8, DT_FLOAT, DT_FLOAT})
      .AddIntArg("T", DT_UINT8)
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();

  auto output = net.GetTensor("Output");
  auto output_min = net.GetTensor("OutputMin");
  auto output_max = net.GetTensor("OutputMax");

  auto expected_output =
      CreateTensor<uint8_t>({1, 2, 3, 1}, {32, 64, 127, 159, 191, 223});
  auto expected_min = CreateTensor<float>({1}, {-3.01887});
  auto expected_max = CreateTensor<float>({1}, {5});

  ExpectTensorNear<uint8_t>(*expected_output, *output);
  ExpectTensorNear<float>(*expected_min, *output_min);
  ExpectTensorNear<float>(*expected_max, *output_max);
}

TEST_F(QuantizeTest, TestQuantizeTrend) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddRandomInput<CPU, float>("Input", {100});
  const float *input_data = net.GetTensor("Input")->data<float>();
  net.AddInputFromArray<CPU, float>(
      "InputMin", {1},
      {*std::min_element(input_data,
                         input_data + net.GetTensor("Input")->size())});
  net.AddInputFromArray<CPU, float>(
      "InputMax", {1},
      {*std::max_element(input_data,
                         input_data + net.GetTensor("Input")->size())});

  OpDefBuilder("Quantize", "QuantizeTest")
      .Input("Input")
      .Input("InputMin")
      .Input("InputMax")
      .Output("Output")
      .Output("OutputMin")
      .Output("OutputMax")
      .OutputType({DT_UINT8, DT_FLOAT, DT_FLOAT})
      .AddIntArg("T", DT_UINT8)
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();

  auto output = net.GetTensor("Output");

  const uint8_t *output_data = net.GetTensor("Output")->data<uint8_t>();
  for (int i = 1; i < output->size(); ++i) {
    if (input_data[i] > input_data[i - 1]) {
      EXPECT_GE(output_data[i], output_data[i - 1]);
    } else if (input_data[i] == input_data[i - 1]) {
      EXPECT_EQ(output_data[i], output_data[i - 1]);
    } else {
      EXPECT_LE(output_data[i], output_data[i - 1]);
    }
  }
}

TEST_F(QuantizeTest, TestDequantize) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<CPU, uint8_t>("Input", {1, 2, 3, 1},
                                      {32, 64, 127, 159, 191, 223});
  net.AddInputFromArray<CPU, float>("InputMin", {1}, {-3.01887});
  net.AddInputFromArray<CPU, float>("InputMax", {1}, {5});

  OpDefBuilder("Dequantize", "DequantizeTest")
      .Input("Input")
      .Input("InputMin")
      .Input("InputMax")
      .Output("Output")
      .OutputType({DT_FLOAT})
      .AddIntArg("T", DT_UINT8)
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();

  auto output = net.GetTensor("Output");
  auto expected_output =
      CreateTensor<float>({1, 2, 3, 1}, {-2, -1, 1, 2, 3, 4});
  auto expected_min = CreateTensor<float>({1}, {-3.01887});
  auto expected_max = CreateTensor<float>({1}, {5});

  ExpectTensorNear<float>(*expected_output, *output, 0.1, 0.01);
}

TEST_F(QuantizeTest, TestRequantizeWithMinMax) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<CPU, int>(
      "Input", {1, 2, 3, 1},
      {-1073741824, -536870912, 536870912, 1073741824, 1610612736, 2147483647});
  net.AddInputFromArray<CPU, float>("InputMin", {1}, {-3});
  net.AddInputFromArray<CPU, float>("InputMax", {1}, {5});
  net.AddInputFromArray<CPU, float>("RerangeMin", {1}, {-3.01887});
  net.AddInputFromArray<CPU, float>("RerangeMax", {1}, {5});

  OpDefBuilder("Requantize", "RequantizeTest")
      .Input("Input")
      .Input("InputMin")
      .Input("InputMax")
      .Input("RerangeMin")
      .Input("RerangeMax")
      .Output("Output")
      .Output("OutputMin")
      .Output("OutputMax")
      .OutputType({DT_UINT8, DT_FLOAT, DT_FLOAT})
      .AddIntArg("T", DT_UINT8)
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();

  auto output = net.GetTensor("Output");
  auto expected_output =
      CreateTensor<uint8_t>({1, 2, 3, 1}, {32, 64, 128, 160, 191, 223});
  auto expected_min = CreateTensor<float>({1}, {-3.01887});
  auto expected_max = CreateTensor<float>({1}, {5});

  ExpectTensorNear<uint8_t>(*expected_output, *output);
}

TEST_F(QuantizeTest, TestRequantizeWithoutMinMax) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<CPU, int>(
      "Input", {1, 2, 3, 1},
      {-1073741824, -536870912, 536870912, 1073741824, 1610612736, 2147483647});
  net.AddInputFromArray<CPU, float>("InputMin", {1}, {-3});
  net.AddInputFromArray<CPU, float>("InputMax", {1}, {5});

  OpDefBuilder("Requantize", "RequantizeTest")
      .Input("Input")
      .Input("InputMin")
      .Input("InputMax")
      .Output("Output")
      .Output("OutputMin")
      .Output("OutputMax")
      .OutputType({DT_UINT8, DT_FLOAT, DT_FLOAT})
      .AddIntArg("T", DT_UINT8)
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();

  auto output = net.GetTensor("Output");
  auto expected_output =
      CreateTensor<uint8_t>({1, 2, 3, 1}, {0, 43, 128, 170, 213, 255});
  auto expected_min = CreateTensor<float>({1}, {-3.01887});
  auto expected_max = CreateTensor<float>({1}, {5});
  ExpectTensorNear<uint8_t>(*expected_output, *output);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
