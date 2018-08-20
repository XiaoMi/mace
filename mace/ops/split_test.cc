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

#include <functional>
#include <vector>

#include "gmock/gmock.h"
#include "mace/ops/ops_test_util.h"
#include "mace/ops/split.h"

namespace mace {
namespace ops {
namespace test {

class SplitOpTest : public OpsTestBase {};

namespace {
template <DeviceType D, typename T>
void RandomTest(const int num_outputs, const int axis) {
  static unsigned int seed = time(NULL);
  const index_t output_channels = 4 * (1 + rand_r(&seed) % 10);
  const index_t input_channels = num_outputs * output_channels;
  const index_t batch = 3 + (rand_r(&seed) % 10);
  const index_t height = 13 + (rand_r(&seed) % 10);
  const index_t width = 17 + (rand_r(&seed) % 10);

  // Construct graph
  OpsTestNet net;

  std::vector<index_t> input_shape;
  if (axis == 1)
    input_shape = {batch, input_channels, height, width};
  else if (axis == 3)
    input_shape = {batch, height, width, input_channels};
  const index_t input_size = std::accumulate(
      input_shape.begin(), input_shape.end(), 1, std::multiplies<index_t>());
  std::vector<float> input_data(input_size);
  GenerateRandomRealTypeData(input_shape, &input_data);
  net.AddInputFromArray<D, float>("Input", input_shape, input_data);

  if (D == DeviceType::GPU) {
    BufferToImage<D, T>(&net, "Input", "InputImage",
                        kernels::BufferType::IN_OUT_CHANNEL);

    auto builder = OpDefBuilder("Split", "SplitTest");
    builder.Input("InputImage");
    for (int i = 0; i < num_outputs; ++i) {
      builder = builder.Output(MakeString("OutputImage", i));
    }
    builder.AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());
  } else {
    auto builder = OpDefBuilder("Split", "SplitTest").AddIntArg("axis", axis);
    builder.Input("Input");
    for (int i = 0; i < num_outputs; ++i) {
      builder = builder.Output(MakeString("Output", i));
    }
    builder.Finalize(net.NewOperatorDef());
  }

  // Run
  net.RunOp(D);

  if (D == DeviceType::GPU) {
    for (int i = 0; i < num_outputs; ++i) {
      ImageToBuffer<D, float>(&net, MakeString("OutputImage", i),
                              MakeString("Output", i),
                              kernels::BufferType::IN_OUT_CHANNEL);
    }
  }

  // Check
  std::vector<index_t> expected_shape;
  if (axis == 1)
    expected_shape = {batch, output_channels, height, width};
  else if (axis == 3)
    expected_shape = {batch, height, width, output_channels};
  const index_t outer_size =
      std::accumulate(expected_shape.begin(), expected_shape.begin() + axis, 1,
                      std::multiplies<index_t>());
  const index_t inner_size =
      std::accumulate(expected_shape.begin() + axis + 1, expected_shape.end(),
                      1, std::multiplies<index_t>());
  const float *input_ptr = input_data.data();
  const float *output_ptr;
  for (int i = 0; i < num_outputs; ++i) {
    auto output = net.GetOutput(MakeString("Output", i).c_str());
    EXPECT_THAT(output->shape(), ::testing::ContainerEq(expected_shape));
    Tensor::MappingGuard output_mapper(output);
    output_ptr = output->data<float>();
    for (int outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
      const int idx =
          (outer_idx * input_channels + i * output_channels) * inner_size;
      for (int j = 0; j < output_channels * inner_size; ++j) {
        ASSERT_NEAR(*output_ptr++, input_ptr[idx + j], 1e-2)
            << "with output " << i << " index " << idx + j;
      }
    }
  }
}
}  // namespace

TEST_F(SplitOpTest, CPU) {
  RandomTest<DeviceType::CPU, float>(2, 3);
  RandomTest<DeviceType::CPU, float>(4, 3);
  RandomTest<DeviceType::CPU, float>(11, 3);
}

TEST_F(SplitOpTest, CPUAxis1) {
  RandomTest<DeviceType::CPU, float>(2, 1);
  RandomTest<DeviceType::CPU, float>(4, 1);
  RandomTest<DeviceType::CPU, float>(11, 1);
}

TEST_F(SplitOpTest, OPENCLFloat) {
  RandomTest<DeviceType::GPU, float>(2, 3);
  RandomTest<DeviceType::GPU, float>(4, 3);
  RandomTest<DeviceType::GPU, float>(11, 3);
}

TEST_F(SplitOpTest, OPENCLHalf) {
  RandomTest<DeviceType::GPU, half>(2, 3);
  RandomTest<DeviceType::GPU, half>(4, 3);
  RandomTest<DeviceType::GPU, half>(11, 3);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
