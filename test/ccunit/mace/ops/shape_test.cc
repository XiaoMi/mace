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

#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class ShapeOpTest : public OpsTestBase {};

namespace {

void TestShapeOp(const std::vector<index_t> &input_shape) {
  OpsTestNet net;
  net.AddRandomInput<CPU, float>("Input", input_shape);
  OpDefBuilder("Shape", "ShapeOpTest")
      .Input("Input")
      .Output("Output")
      .OutputType({DataTypeToEnum<int32_t>::v()})
      .Finalize(net.NewOperatorDef());

  net.RunOp();

  // we need to convert vector<index_t> to vector<int32_t>
  std::vector<int32_t> expected_input_shape(input_shape.begin(),
                                            input_shape.end());
  if (!expected_input_shape.empty()) {
    net.AddInputFromArray<CPU, int32_t>("ExpectedOutput",
                                        {static_cast<int32_t>(
                                             input_shape.size())},
                                        expected_input_shape);
  } else {
    net.AddInputFromArray<CPU, int32_t>("ExpectedOutput", {}, {0});
  }

  ExpectTensorNear<int32_t>(*net.GetOutput("ExpectedOutput"),
                            *net.GetOutput("Output"));
}

}  // namespace

TEST_F(ShapeOpTest, TestShape) {
  TestShapeOp({1, 2, 3});
  TestShapeOp({2, 3});
  TestShapeOp({3});
  TestShapeOp({});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
