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

#include "gmock/gmock.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class ExpandDimsTest : public OpsTestBase {};

namespace {
template <DeviceType D, typename T>
void TestExpandDims(const std::vector<index_t> &input_shape,
                    const int &axis,
                    const std::vector<index_t> &output_shape) {
  // Construct graph
  OpsTestNet net;
  OpDefBuilder("ExpandDims", "ExpandDimsTest")
      .Input("Input")
      .AddIntArg("axis", static_cast<int>(axis))
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddRandomInput<D, T>("Input", input_shape);

  // Run
  net.RunOp();

  auto input = net.GetTensor("Input");
  auto output = net.GetTensor("Output");

  EXPECT_THAT(output->shape(), ::testing::ContainerEq(output_shape));

  const T *input_ptr = input->data<T>();
  const T *output_ptr = output->data<T>();
  const int size = output->size();
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(input_ptr[i], output_ptr[i]);
  }
}
}  // namespace

TEST_F(ExpandDimsTest, SimpleCPU) {
  TestExpandDims<DeviceType::CPU, float>({3, 2, 1}, 1, {3, 1, 2, 1});
  TestExpandDims<DeviceType::CPU, float>({1, 2, 3}, -1, {1, 2, 3, 1});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
