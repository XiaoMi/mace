//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/concat.h"
#include "gmock/gmock.h"
#include "mace/ops/ops_test_util.h"

using namespace mace;

class ConcatOpTest : public OpsTestBase {};

TEST_F(ConcatOpTest, Simple_Horizon) {
  // Construct graph
  auto &net = test_net();
  OpDefBuilder("Concat", "ConcatTest")
      .Input("Input0")
      .Input("Input1")
      .Input("Axis")
      .Output("Output")
      .Finalize(net.new_operator_def());

  std::vector<index_t> input_shape = {4, 4};
  std::vector<float> input0;
  GenerateRandomRealTypeData(input_shape, input0);
  std::vector<float> input1;
  GenerateRandomRealTypeData(input_shape, input1);
  // Add inputs
  net.AddInputFromArray<DeviceType::CPU, float>("Input0", input_shape, input0);
  net.AddInputFromArray<DeviceType::CPU, float>("Input1", input_shape, input1);
  net.AddInputFromArray<DeviceType::CPU, int>("Axis", {}, {0});

  // Run
  net.RunOp();

  // Check
  auto output = net.GetOutput("Output");

  std::vector<index_t> expected_shape = {8, 4};
  EXPECT_THAT(output->shape(), ::testing::ContainerEq(expected_shape));

  const float *output_ptr = output->data<float>();
  for (auto f : input0) {
    ASSERT_EQ(f, *output_ptr++);
  }
  for (auto f : input1) {
    ASSERT_EQ(f, *output_ptr++);
  }
}

TEST_F(ConcatOpTest, Simple_Vertical) {
  // Construct graph
  auto &net = test_net();
  OpDefBuilder("Concat", "ConcatTest")
      .Input("Input0")
      .Input("Input1")
      .Input("Axis")
      .Output("Output")
      .Finalize(net.new_operator_def());

  std::vector<index_t> input_shape = {4, 4};
  std::vector<float> input0;
  GenerateRandomRealTypeData(input_shape, input0);
  std::vector<float> input1;
  GenerateRandomRealTypeData(input_shape, input1);
  // Add inputs
  net.AddInputFromArray<DeviceType::CPU, float>("Input0", input_shape, input0);
  net.AddInputFromArray<DeviceType::CPU, float>("Input1", input_shape, input1);
  net.AddInputFromArray<DeviceType::CPU, int>("Axis", {}, {1});

  // Run
  net.RunOp();

  // Check
  auto output = net.GetOutput("Output");

  std::vector<index_t> expected_shape = {4, 8};
  EXPECT_THAT(output->shape(), ::testing::ContainerEq(expected_shape));

  const float *output_ptr = output->data<float>();
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      ASSERT_EQ(input0[i * 4 + j], *output_ptr++);
    }
    for (int j = 0; j < 4; ++j) {
      ASSERT_EQ(input1[i * 4 + j], *output_ptr++);
    }
  }
}

TEST_F(ConcatOpTest, Random) {
  srand(time(nullptr));
  int dim = 5;
  int num_inputs = 2 + rand() % 10;
  int axis = rand() % dim;
  // Construct graph
  auto &net = test_net();
  auto builder = OpDefBuilder("Concat", "ConcatTest");
  for (int i = 0; i < num_inputs; ++i) {
    builder = builder.Input(("Input" + ToString(i)).c_str());
  }
  builder.Input("Axis").Output("Output").Finalize(net.new_operator_def());

  std::vector<index_t> shape_data;
  GenerateRandomIntTypeData<index_t>({dim}, shape_data, 1, dim);
  std::vector<std::vector<index_t>> input_shapes(num_inputs, shape_data);
  std::vector<std::vector<float>> inputs(num_inputs, std::vector<float>());
  std::vector<float *> input_ptrs(num_inputs, nullptr);
  index_t concat_axis_size = 0;
  for (int i = 0; i < num_inputs; ++i) {
    input_shapes[i][axis] = 1 + rand() % dim;
    concat_axis_size += input_shapes[i][axis];
    GenerateRandomRealTypeData(input_shapes[i], inputs[i]);
    input_ptrs[i] = inputs[i].data();
    net.AddInputFromArray<DeviceType::CPU, float>(("Input" + ToString(i)).c_str(),
                                 input_shapes[i], inputs[i]);
  }
  net.AddInputFromArray<DeviceType::CPU, int>("Axis", {}, {axis});

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
