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
#include "mace/kernels/conv_pool_2d_util.h"

namespace mace {
namespace ops {
namespace test {

class InferConv2dShapeOpTest : public OpsTestBase {};

namespace {

void TestInferConv2dShapeOp(const std::vector<index_t> &input_shape,
                            const int stride,
                            const std::vector<index_t> &output_shape) {
  OpsTestNet net;
  net.AddRandomInput<CPU, float>("Input", input_shape);
  const int in_ch = static_cast<int>(input_shape[3]);
  const int out_ch = static_cast<int>(output_shape[3]);
  OpDefBuilder("InferConv2dShape", "InferConv2dShapeOpTest")
      .Input("Input")
      .Output("Output")
      .AddIntArg("datd_format", 0)
      .AddIntsArg("strides", {stride, stride})
      .AddIntsArg("kernels", {out_ch, in_ch, 3, 3})
      .AddIntArg("padding", Padding::SAME)
      .OutputType({DataTypeToEnum<int32_t>::v()})
      .Finalize(net.NewOperatorDef());

  net.RunOp();

  std::vector<int32_t> expected_output_shape(output_shape.begin(),
                                             output_shape.end());
  net.AddInputFromArray<CPU, int32_t>("ExpectedOutput",
                                      {static_cast<int32_t>(
                                           output_shape.size())},
                                      expected_output_shape);


  ExpectTensorNear<int32_t>(*net.GetOutput("ExpectedOutput"),
                            *net.GetOutput("Output"));
}

}  // namespace

TEST_F(InferConv2dShapeOpTest, TestInferConv2dShape) {
TestInferConv2dShapeOp({3, 640, 480, 16}, 1, {3, 640, 480, 3});
TestInferConv2dShapeOp({3, 640, 480, 16}, 2, {3, 320, 240, 3});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
