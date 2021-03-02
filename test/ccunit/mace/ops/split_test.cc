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

#include <functional>
#include <vector>

#include "gmock/gmock.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class SplitOpTest : public OpsTestBase {};

namespace {
template <RuntimeType D, typename T>
void RandomTest(const int num_outputs, int axis) {
  static unsigned int seed = time(NULL);
  const index_t output_channels = 4 * (1 + rand_r(&seed) % 10);
  const index_t input_channels = num_outputs * output_channels;
  const index_t batch = 3 + (rand_r(&seed) % 10);
  const index_t height = 13 + (rand_r(&seed) % 10);
  const index_t width = 17 + (rand_r(&seed) % 10);

  // Construct graph
  OpsTestNet net;

  std::vector<index_t> input_shape;
  if (D == RuntimeType::RT_CPU)
    input_shape = {batch, input_channels, height, width};
  else
    input_shape = {batch, height, width, input_channels};
  const index_t input_size = std::accumulate(
      input_shape.begin(), input_shape.end(), 1, std::multiplies<index_t>());
  std::vector<float> input_data(input_size);
  GenerateRandomRealTypeData(input_shape, &input_data);
  net.AddInputFromArray<D, float>("Input", input_shape, input_data);

  auto builder = OpDefBuilder("Split", "SplitTest").AddIntArg("axis", axis);
  builder.Input("Input");
  for (int i = 0; i < num_outputs; ++i) {
    builder = builder.Output(MakeString("Output", i));
  }
  builder.AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .AddIntArg("has_data_format", 1)
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(D);

  // Check
  std::vector<index_t> expected_shape;
  if (D == RuntimeType::RT_CPU) {
    if (axis == 3) axis = 1;
    expected_shape = {batch, output_channels, height, width};
  } else {
    expected_shape = {batch, height, width, output_channels};
  }
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
    Tensor::MappingGuard output_mapper(output);
    EXPECT_THAT(output->shape(), ::testing::ContainerEq(expected_shape));
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

template <RuntimeType D, typename T>
void RandomTestSizeSplits(const int input_channels,
                          std::vector<int32_t> size_splits,
                          int axis) {
  static unsigned int seed = time(NULL);
  const index_t batch = 3 + (rand_r(&seed) % 10);
  const index_t height = 13 + (rand_r(&seed) % 10);
  const index_t width = 17 + (rand_r(&seed) % 10);

  OpsTestNet net;
  std::vector<index_t> input_shape;
  const std::vector<index_t> size_splits_shape =
      {static_cast<index_t>(size_splits.size())};
  if (D == RuntimeType::RT_CPU)
    input_shape = {batch, input_channels, height, width};
  else
    input_shape = {batch, height, width, input_channels};
  const index_t input_size = std::accumulate(
      input_shape.begin(), input_shape.end(), 1, std::multiplies<index_t>());
  std::vector<float> input_data(input_size);
  GenerateRandomRealTypeData(input_shape, &input_data);
  net.AddInputFromArray<D, float>("Input", input_shape, input_data);
  net.AddInputFromArray<D, int32_t>(
      "InputSizeSplits", size_splits_shape, size_splits);
  auto builder = OpDefBuilder("Split", "SplitTest").AddIntArg("axis", axis);
  builder.Input("Input");
  builder.Input("InputSizeSplits");
  for (size_t i = 0; i < size_splits.size(); ++i) {
    builder = builder.Output(MakeString("Output", i));
  }
  builder.AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .AddIntArg("has_data_format", 1)
      .Finalize(net.NewOperatorDef());
  // Run
  net.RunOp(D);
  std::vector<std::vector<index_t>> expected_shape_list;
  if (axis == 3) {
    axis = 1;
  } else if (axis == 2) {
    axis = 3;
  } else if (axis == 1) {
    axis = 2;
  }
  for (size_t i = 0; i < size_splits.size(); ++i) {
    if (D == RuntimeType::RT_CPU) {
      expected_shape_list.push_back({batch, size_splits[i], height, width});
    } else {
      expected_shape_list.push_back({batch, height, width, size_splits[i]});
    }
  }
  const std::vector<index_t> &expected_shape = expected_shape_list[0];
  const index_t outer_size =
      std::accumulate(expected_shape.begin(), expected_shape.begin() + axis, 1,
                      std::multiplies<index_t>());
  const index_t inner_size =
      std::accumulate(expected_shape.begin() + axis + 1, expected_shape.end(),
                      1, std::multiplies<index_t>());
  const float *input_ptr = input_data.data();
  const float *output_ptr = nullptr;
  std::vector<int> previous_channels(size_splits.size(), 0);
  for (size_t i = 1; i < size_splits.size(); ++i) {
    previous_channels[i] = previous_channels[i-1] + size_splits[i-1];
  }
  for (size_t i = 0; i < size_splits.size(); ++i) {
    auto output = net.GetOutput(MakeString("Output", i).c_str());
    EXPECT_THAT(output->shape(),
                ::testing::ContainerEq(expected_shape_list[i]));
    output_ptr = output->data<float>();
    const float *input_base_ptr = input_ptr + previous_channels[i] * inner_size;
    for (int outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
      for (int j = 0; j  < size_splits[i] * inner_size; ++j) {
        index_t offset = outer_idx * size_splits[i] * inner_size + j;
        ASSERT_NEAR(output_ptr[offset], input_base_ptr[j], 1e-2)
            << "with output " << i << " index " << offset;
      }
      input_base_ptr += input_channels * inner_size;
    }
  }
}
}  // namespace

TEST_F(SplitOpTest, RT_CPU) {
  RandomTest<RuntimeType::RT_CPU, float>(2, 3);
  RandomTest<RuntimeType::RT_CPU, float>(4, 3);
  RandomTest<RuntimeType::RT_CPU, float>(11, 3);
}

TEST_F(SplitOpTest, CPUSizeSplits) {
  RandomTestSizeSplits<RuntimeType::RT_CPU, float>(4, {1, 2, 1}, 3);
  RandomTestSizeSplits<RuntimeType::RT_CPU, float>(
      96, {10, 14, 9, 15, 8, 16, 7, 17}, 3);
  RandomTestSizeSplits<RuntimeType::RT_CPU, float>(10, {8, 2}, 3);
}

TEST_F(SplitOpTest, CPUAxis1) {
  RandomTest<RuntimeType::RT_CPU, float>(2, 3);
  RandomTest<RuntimeType::RT_CPU, float>(4, 3);
  RandomTest<RuntimeType::RT_CPU, float>(11, 3);
}

TEST_F(SplitOpTest, OPENCLFloat) {
  RandomTest<RuntimeType::RT_OPENCL, float>(2, 3);
  RandomTest<RuntimeType::RT_OPENCL, float>(4, 3);
  RandomTest<RuntimeType::RT_OPENCL, float>(11, 3);
}

TEST_F(SplitOpTest, OPENCLHalf) {
  RandomTest<RuntimeType::RT_OPENCL, half>(2, 3);
  RandomTest<RuntimeType::RT_OPENCL, half>(4, 3);
  RandomTest<RuntimeType::RT_OPENCL, half>(11, 3);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
