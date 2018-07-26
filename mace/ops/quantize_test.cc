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

namespace {

void TestQuantizeDequantize(const std::vector<float> &input, bool non_zero) {
  OpsTestNet net;
  net.AddInputFromArray<CPU, float>("Input",
                                    {static_cast<index_t>(input.size())},
                                    input);
  OpDefBuilder("Quantize", "QuantizeTest")
      .Input("Input")
      .Output("QuantizeOutput")
      .OutputType({DT_UINT8})
      .AddIntArg("T", DT_UINT8)
      .AddIntArg("non_zero", non_zero)
      .Finalize(net.NewOperatorDef());

  net.RunOp();

  if (non_zero) {
    Tensor *quantized_output = net.GetTensor("QuantizeOutput");
    Tensor::MappingGuard guard(quantized_output);
    const uint8_t *quantized_output_data = quantized_output->data<uint8_t>();
    for (index_t i = 0; i < quantized_output->size(); ++i) {
      EXPECT_GT(quantized_output_data[i], 0);
    }
  }

  OpDefBuilder("Dequantize", "DeQuantizeTest")
      .Input("QuantizeOutput")
      .Output("Output")
      .OutputType({DT_FLOAT})
      .AddIntArg("T", DT_UINT8)
      .Finalize(net.NewOperatorDef());

  net.RunOp();

  auto output = net.GetTensor("Output");

  ExpectTensorNear<float>(*net.GetTensor("Input"),
                          *net.GetTensor("Output"),
                          0.1);
}

}  // namespace

class QuantizeTest : public OpsTestBase {};

TEST_F(QuantizeTest, TestQuantize) {
  TestQuantizeDequantize({-2, -1, 0, 1, 2, 3, 4}, false);
  TestQuantizeDequantize({-2, -1, 0, 1, 2, 3, 4}, true);
  TestQuantizeDequantize({0, 1, 2, 3, 4}, false);
  TestQuantizeDequantize({0, 1, 2, 3, 4}, true);
  TestQuantizeDequantize({2, 3, 4, 5, 6}, false);
  TestQuantizeDequantize({2, 3, 4, 5, 6}, true);
  TestQuantizeDequantize({2, 4, 6, 8}, false);
  TestQuantizeDequantize({2, 4, 6, 8}, true);
  TestQuantizeDequantize({-2, -4, -6, -8}, false);
  TestQuantizeDequantize({-2, -4, -6, -8}, true);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
