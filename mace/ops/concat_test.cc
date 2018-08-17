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
#include <string>

#include "gmock/gmock.h"
#include "mace/ops/concat.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class ConcatOpTest : public OpsTestBase {};

TEST_F(ConcatOpTest, CPUSimpleHorizon) {
  // Construct graph
  OpsTestNet net;
  OpDefBuilder("Concat", "ConcatTest")
      .Input("Input0")
      .Input("Input1")
      .AddIntArg("axis", 0)
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  std::vector<index_t> input_shape = {4, 4};
  std::vector<float> input0;
  GenerateRandomRealTypeData(input_shape, &input0);
  std::vector<float> input1;
  GenerateRandomRealTypeData(input_shape, &input1);
  // Add inputs
  net.AddInputFromArray<DeviceType::CPU, float>("Input0", input_shape, input0);
  net.AddInputFromArray<DeviceType::CPU, float>("Input1", input_shape, input1);

  // Run
  net.RunOp();

  // Check
  auto output = net.GetOutput("Output");

  std::vector<index_t> expected_shape = {8, 4};
  EXPECT_THAT(output->shape(), ::testing::ContainerEq(expected_shape));

  const float *output_ptr = output->data<float>();
  for (auto f : input0) {
    EXPECT_EQ(f, *output_ptr++);
  }
  for (auto f : input1) {
    EXPECT_EQ(f, *output_ptr++);
  }
}

TEST_F(ConcatOpTest, CPUSimpleVertical) {
  // Construct graph
  OpsTestNet net;
  OpDefBuilder("Concat", "ConcatTest")
      .Input("Input0")
      .Input("Input1")
      .AddIntArg("axis", 1)
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  std::vector<index_t> input_shape = {4, 4};
  std::vector<float> input0;
  GenerateRandomRealTypeData(input_shape, &input0);
  std::vector<float> input1;
  GenerateRandomRealTypeData(input_shape, &input1);
  // Add inputs
  net.AddInputFromArray<DeviceType::CPU, float>("Input0", input_shape, input0);
  net.AddInputFromArray<DeviceType::CPU, float>("Input1", input_shape, input1);

  // Run
  net.RunOp();

  // Check
  auto output = net.GetOutput("Output");

  std::vector<index_t> expected_shape = {4, 8};
  EXPECT_THAT(output->shape(), ::testing::ContainerEq(expected_shape));

  const float *output_ptr = output->data<float>();
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_EQ(input0[i * 4 + j], *output_ptr++);
    }
    for (int j = 0; j < 4; ++j) {
      EXPECT_EQ(input1[i * 4 + j], *output_ptr++);
    }
  }
}

TEST_F(ConcatOpTest, CPURandom) {
  static unsigned int seed = time(NULL);
  int dim = 5;
  int num_inputs = 2 + rand_r(&seed) % 10;
  int axis = rand_r(&seed) % dim;
  // Construct graph
  OpsTestNet net;
  auto builder = OpDefBuilder("Concat", "ConcatTest");
  for (int i = 0; i < num_inputs; ++i) {
    builder = builder.Input(MakeString("Input", i));
  }
  builder.AddIntArg("axis", axis)
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  std::vector<index_t> shape_data;
  GenerateRandomIntTypeData<index_t>({dim}, &shape_data, 1, dim);
  std::vector<std::vector<index_t>> input_shapes(num_inputs, shape_data);
  std::vector<std::vector<float>> inputs(num_inputs, std::vector<float>());
  std::vector<float *> input_ptrs(num_inputs, nullptr);
  index_t concat_axis_size = 0;
  for (int i = 0; i < num_inputs; ++i) {
    input_shapes[i][axis] = 1 + rand_r(&seed) % dim;
    concat_axis_size += input_shapes[i][axis];
    GenerateRandomRealTypeData(input_shapes[i], &inputs[i]);
    input_ptrs[i] = inputs[i].data();
    net.AddInputFromArray<DeviceType::CPU, float>(MakeString("Input", i),
                                                  input_shapes[i], inputs[i]);
  }

  // Run
  net.RunOp();

  // Check
  auto output = net.GetOutput("Output");

  std::vector<index_t> expected_shape = input_shapes[0];
  expected_shape[axis] = concat_axis_size;
  EXPECT_THAT(output->shape(), ::testing::ContainerEq(expected_shape));

  const float *output_ptr = output->data<float>();
  while (output_ptr != (output->data<float>() + output->size())) {
    for (int i = 0; i < num_inputs; ++i) {
      index_t num_elements =
          std::accumulate(input_shapes[i].begin() + axis, input_shapes[i].end(),
                          1, std::multiplies<index_t>());
      for (int j = 0; j < num_elements; ++j) {
        EXPECT_EQ(*input_ptrs[i]++, *output_ptr++);
      }
    }
  }
}

namespace {
template <typename T>
void OpenclRandomTest(const std::vector<std::vector<index_t>> &shapes,
                      const int axis) {
  srand(time(nullptr));
  int num_inputs = shapes.size();
  int concat_axis_size = 0;
  // Construct graph
  std::vector<std::vector<float>> inputs(num_inputs, std::vector<float>());
  std::vector<const float *> input_ptrs(num_inputs);
  OpsTestNet net;
  for (int i = 0; i < num_inputs; ++i) {
    const std::string input_name = MakeString("Input", i);
    const std::string image_name = MakeString("InputImage", i);
    concat_axis_size += shapes[i][axis];
    GenerateRandomRealTypeData(shapes[i], &inputs[i]);
    input_ptrs[i] = inputs[i].data();
    net.AddInputFromArray<DeviceType::GPU, float>(input_name, shapes[i],
                                                  inputs[i]);
    BufferToImage<DeviceType::GPU, T>(&net, input_name, image_name,
                                      kernels::BufferType::IN_OUT_CHANNEL);
  }

  auto builder = OpDefBuilder("Concat", "ConcatTest");
  for (int i = 0; i < num_inputs; ++i) {
    const std::string image_name = MakeString("InputImage", i);
    builder = builder.Input(image_name);
  }
  builder.AddIntArg("axis", axis)
      .Output("OutputImage")
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(DeviceType::GPU);

  ImageToBuffer<DeviceType::GPU, float>(&net, "OutputImage", "Output",
                                        kernels::BufferType::IN_OUT_CHANNEL);

  // Check
  auto output = net.GetOutput("Output");

  std::vector<index_t> expected_shape = shapes[0];
  expected_shape[axis] = concat_axis_size;
  EXPECT_THAT(output->shape(), ::testing::ContainerEq(expected_shape));

  Tensor::MappingGuard output_mapper(output);
  const float *output_ptr = output->data<float>();
  const float *output_ptr_end = output_ptr + output->size();
  int k = 0;
  while (output_ptr != output_ptr_end) {
    for (int i = 0; i < num_inputs; ++i) {
      index_t num_elements =
          std::accumulate(shapes[i].begin() + axis, shapes[i].end(), 1,
                          std::multiplies<index_t>());

      const float *input_ptr = input_ptrs[i] + k * num_elements;
      for (int j = 0; j < num_elements; ++j) {
        EXPECT_NEAR(*(input_ptr + j), *output_ptr++, 1e-2)
            << "With index: " << i << ", " << j;
      }
    }
    k++;
  }
}
}  // namespace

TEST_F(ConcatOpTest, OPENCLAligned) {
  OpenclRandomTest<float>({{3, 32, 32, 32}, {3, 32, 32, 64}}, 3);
}

TEST_F(ConcatOpTest, OPENCLHalfAligned) {
  OpenclRandomTest<half>({{3, 32, 32, 32}, {3, 32, 32, 64}}, 3);
}

TEST_F(ConcatOpTest, OPENCLUnAligned) {
  OpenclRandomTest<float>({{3, 32, 32, 13}, {3, 32, 32, 17}}, 3);
}

TEST_F(ConcatOpTest, OPENCLAlignedMultiInput) {
  OpenclRandomTest<float>(
      {{3, 32, 32, 32}, {3, 32, 32, 32}, {3, 32, 32, 32}, {3, 32, 32, 32}}, 3);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
