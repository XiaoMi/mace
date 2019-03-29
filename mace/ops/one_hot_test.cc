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

class OneHotTest : public OpsTestBase {};

namespace {
template <DeviceType D, typename T>
void TestOneHot(const std::vector<index_t> &input_shape,
                const std::vector<float> &input_data,
                const std::vector<index_t> &expected_shape,
                const std::vector<float> &expected_data,
                const int depth,
                const int axis,
                const int on_value = 1,
                const int off_value = 0) {
  // Construct graph
  OpsTestNet net;
  std::string input("Input");
  std::string output("Output");

  // Add input data
  net.AddInputFromArray<D, float>(input, input_shape, input_data);

  OpDefBuilder("OneHot", "OneHotTest")
  .Input(input)
  .Output(output)
  .AddIntArg("depth", depth)
  .AddFloatArg("on_value", on_value)
  .AddFloatArg("off_value", off_value)
  .AddIntArg("axis", axis)
  .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(D);

  auto actual = net.GetTensor(output.c_str());
  auto expected = net.CreateTensor<float>(expected_shape, expected_data);

  ExpectTensorNear<float>(*expected, *actual, 1e-5);
}
}  // namespace

TEST_F(OneHotTest, Dim1) {
  const std::vector<index_t> input_shape{10};
  const std::vector<float>   input_data{1, 3, 1, 8, 3, 2, 2, 3, 1, 2};
  std::vector<index_t>       expected_shape{10, 5};
  std::vector<float>         expected_data{
    0, 1, 0, 0, 0,
    0, 0, 0, 1, 0,
    0, 1, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 1, 0,
    0, 0, 1, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 0, 1, 0,
    0, 1, 0, 0, 0,
    0, 0, 1, 0, 0,
  };

  TestOneHot<DeviceType::CPU, float>(input_shape, input_data, expected_shape,
                                     expected_data, 5, -1);

  expected_shape = {5, 10};
  expected_data  = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 1, 1, 0, 0, 1,
    0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  TestOneHot<DeviceType::CPU, float>(input_shape, input_data, expected_shape,
                                     expected_data, 5, 0);
}

TEST_F(OneHotTest, OnOffValue) {
  const std::vector<index_t> input_shape{3};
  const std::vector<float>   input_data{0, 2, 5};
  const std::vector<index_t> expected_shape{3, 6};
  const std::vector<float>   expected_data{
    7, 8, 8, 8, 8, 8,
    8, 8, 7, 8, 8, 8,
    8, 8, 8, 8, 8, 7,
  };

  TestOneHot<DeviceType::CPU, float>(input_shape, input_data, expected_shape,
                                     expected_data, 6, -1, 7, 8);
}

TEST_F(OneHotTest, Dim2) {
  const std::vector<index_t> input_shape{2, 3};
  const std::vector<float>   input_data{
    1, 3, 2,
    0, 1, 1,
  };
  std::vector<index_t>       expected_shape{4, 2, 3};
  std::vector<float>         expected_data{
    0, 0, 0,
    1, 0, 0,

    1, 0, 0,
    0, 1, 1,

    0, 0, 1,
    0, 0, 0,

    0, 1, 0,
    0, 0, 0,
  };

  TestOneHot<DeviceType::CPU, float>(input_shape, input_data, expected_shape,
                                     expected_data, 4, 0);

  expected_shape = {2, 4, 3};
  expected_data = {
    0, 0, 0,
    1, 0, 0,
    0, 0, 1,
    0, 1, 0,

    1, 0, 0,
    0, 1, 1,
    0, 0, 0,
    0, 0, 0,
  };

  TestOneHot<DeviceType::CPU, float>(input_shape, input_data, expected_shape,
                                     expected_data, 4, 1);

  expected_shape = {2, 3, 4};
  expected_data = {
    0, 1, 0, 0,
    0, 0, 0, 1,
    0, 0, 1, 0,

    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 1, 0, 0,
  };

  TestOneHot<DeviceType::CPU, float>(input_shape, input_data, expected_shape,
                                     expected_data, 4, 2);
}

TEST_F(OneHotTest, Dim3) {
  const std::vector<index_t> input_shape{2, 3, 4};
  const std::vector<float>   input_data{
    3, 1, 3, 0,
    0, 1, 3, 1,
    2, 2, 1, 0,

    1, 2, 0, 1,
    3, 2, 1, 1,
    0, 1, 3, 0,
  };
  std::vector<index_t>       expected_shape{4, 2, 3, 4};
  std::vector<float>         expected_data{
    0, 0, 0, 1,
    1, 0, 0, 0,
    0, 0, 0, 1,

    0, 0, 1, 0,
    0, 0, 0, 0,
    1, 0, 0, 1,


    0, 1, 0, 0,
    0, 1, 0, 1,
    0, 0, 1, 0,

    1, 0, 0, 1,
    0, 0, 1, 1,
    0, 1, 0, 0,


    0, 0, 0, 0,
    0, 0, 0, 0,
    1, 1, 0, 0,

    0, 1, 0, 0,
    0, 1, 0, 0,
    0, 0, 0, 0,


    1, 0, 1, 0,
    0, 0, 1, 0,
    0, 0, 0, 0,

    0, 0, 0, 0,
    1, 0, 0, 0,
    0, 0, 1, 0,
  };

  TestOneHot<DeviceType::CPU, float>(input_shape, input_data, expected_shape,
                                     expected_data, 4, 0);

  expected_shape = {2, 4, 3, 4};
  expected_data = {
    0, 0, 0, 1,
    1, 0, 0, 0,
    0, 0, 0, 1,

    0, 1, 0, 0,
    0, 1, 0, 1,
    0, 0, 1, 0,

    0, 0, 0, 0,
    0, 0, 0, 0,
    1, 1, 0, 0,

    1, 0, 1, 0,
    0, 0, 1, 0,
    0, 0, 0, 0,


    0, 0, 1, 0,
    0, 0, 0, 0,
    1, 0, 0, 1,

    1, 0, 0, 1,
    0, 0, 1, 1,
    0, 1, 0, 0,

    0, 1, 0, 0,
    0, 1, 0, 0,
    0, 0, 0, 0,

    0, 0, 0, 0,
    1, 0, 0, 0,
    0, 0, 1, 0,
  };

  TestOneHot<DeviceType::CPU, float>(input_shape, input_data, expected_shape,
                                     expected_data, 4, 1);

  expected_shape = {2, 3, 4, 4};
  expected_data = {
    0, 0, 0, 1,
    0, 1, 0, 0,
    0, 0, 0, 0,
    1, 0, 1, 0,

    1, 0, 0, 0,
    0, 1, 0, 1,
    0, 0, 0, 0,
    0, 0, 1, 0,

    0, 0, 0, 1,
    0, 0, 1, 0,
    1, 1, 0, 0,
    0, 0, 0, 0,


    0, 0, 1, 0,
    1, 0, 0, 1,
    0, 1, 0, 0,
    0, 0, 0, 0,

    0, 0, 0, 0,
    0, 0, 1, 1,
    0, 1, 0, 0,
    1, 0, 0, 0,

    1, 0, 0, 1,
    0, 1, 0, 0,
    0, 0, 0, 0,
    0, 0, 1, 0,
  };

  TestOneHot<DeviceType::CPU, float>(input_shape, input_data, expected_shape,
                                     expected_data, 4, 2);

  expected_shape = {2, 3, 4, 4};
  expected_data = {
    0, 0, 0, 1,
    0, 1, 0, 0,
    0, 0, 0, 1,
    1, 0, 0, 0,

    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 0, 1,
    0, 1, 0, 0,

    0, 0, 1, 0,
    0, 0, 1, 0,
    0, 1, 0, 0,
    1, 0, 0, 0,


    0, 1, 0, 0,
    0, 0, 1, 0,
    1, 0, 0, 0,
    0, 1, 0, 0,

    0, 0, 0, 1,
    0, 0, 1, 0,
    0, 1, 0, 0,
    0, 1, 0, 0,

    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 0, 1,
    1, 0, 0, 0,
  };

  TestOneHot<DeviceType::CPU, float>(input_shape, input_data, expected_shape,
                                     expected_data, 4, 3);
}

TEST_F(OneHotTest, CPUFallback) {
  for (int dim = 1; dim < 7; ++dim) {
    std::vector<index_t> shape_in(dim, 1);
    std::vector<index_t> shape_out(dim + 1, 1);
    OpsTestNet net;

    net.AddRepeatedInput<DeviceType::GPU, float>("Input", shape_in, 0);

    OpDefBuilder("OneHot", "OneHotTest")
    .Input("Input")
    .Output("Output")
    .OutputShape(shape_out)
    .AddIntArg("depth", 1)
    .Finalize(net.NewOperatorDef());

    net.RunOp(DeviceType::GPU);
  }
}

}  // namespace test
}  // namespace ops
}  // namespace mace
