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

#include "mace/libmace/mace_api_test.h"

namespace mace {
namespace test {

class MaceAPITest  : public ::testing::Test {};

namespace {


// The height and width of input and output must be equal.
template <DeviceType D, typename T>
void MaceRun(const int in_out_size,
             const std::vector<int64_t> &max_shape,
             const std::vector<std::vector<int64_t>> &input_shapes,
             const std::vector<std::vector<int64_t>> &output_shapes,
             const std::vector<int64_t> &filter_shape) {
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  for (int i = 0; i < in_out_size; ++i) {
    input_names.push_back(MakeString("input", i));
    output_names.push_back(MakeString("output", i));
  }
  std::string filter_tensor_name = "filter";

  std::shared_ptr<NetDef> net_def(new NetDef());

  std::vector<T> data;
  ops::test::GenerateRandomRealTypeData<T>(filter_shape, &data);
  AddTensor<T>(filter_tensor_name, filter_shape, 0, data.size(), net_def.get());

  for (size_t i = 0; i < input_names.size(); ++i) {
    InputOutputInfo *info = net_def->add_input_info();
    info->set_data_format(static_cast<int>(DataFormat::NHWC));
    info->set_name(input_names[i]);
    for (auto d : max_shape) {
      info->add_dims(static_cast<int>(d));
    }
  }
  for (size_t i = 0; i < output_names.size(); ++i) {
    InputOutputInfo *info = net_def->add_output_info();
    info->set_name(output_names[i]);
  }
  for (size_t i = 0; i < output_names.size(); ++i) {
    Conv3x3<T>(input_names[i], filter_tensor_name,
               output_names[i], max_shape,
               net_def.get());
  }

  MaceEngineConfig config(D);

  MaceEngine engine(config);
  MaceStatus status = engine.Init(net_def.get(), input_names, output_names,
      reinterpret_cast<unsigned char *>(data.data()));
  EXPECT_EQ(status, MaceStatus::MACE_SUCCESS);

  std::map<std::string, mace::MaceTensor> inputs;
  std::map<std::string, mace::MaceTensor> outputs;

  for (int i = 0; i < 5; ++i) {
    size_t input_shape_size = input_shapes.size();
    for (size_t j = 0; j < input_shape_size; ++j) {
      inputs.clear();
      outputs.clear();
      GenerateInputs(input_names, input_shapes[j], &inputs);
      GenerateOutputs(output_names, output_shapes[j], &outputs);
      engine.Run(inputs, &outputs);
    }
  }

  CheckOutputs<D, T>(*net_def, inputs, outputs, data);
}

}  // namespace

TEST_F(MaceAPITest, SingleInputOutput) {
  MaceRun<CPU, float>(1,
                      {1, 32, 32, 16},
                      {{1, 32, 32, 16}},
                      {{1, 32, 32, 16}},
                      {16, 16, 3, 3});
  MaceRun<GPU, float>(1,
                      {1, 32, 32, 16},
                      {{1, 32, 32, 16}},
                      {{1, 32, 32, 16}},
                      {16, 16, 3, 3});
  MaceRun<GPU, half>(1,
                     {1, 32, 32, 16},
                     {{1, 32, 32, 16}},
                     {{1, 32, 32, 16}},
                     {16, 16, 3, 3});
}

TEST_F(MaceAPITest, MultipleInputOutput) {
  MaceRun<CPU, float>(2,
                      {1, 16, 32, 16},
                      {{1, 16, 32, 16}},
                      {{1, 16, 32, 16}},
                      {16, 16, 3, 3});
  MaceRun<GPU, float>(2,
                      {1, 16, 32, 16},
                      {{1, 16, 32, 16}},
                      {{1, 16, 32, 16}},
                      {16, 16, 3, 3});
  MaceRun<GPU, half>(2,
                     {1, 16, 32, 16},
                     {{1, 16, 32, 16}},
                     {{1, 16, 32, 16}},
                     {16, 16, 3, 3});
}

TEST_F(MaceAPITest, VariableInputShape) {
  MaceRun<CPU, float>(1,
                      {1, 32, 64, 16},
                      {{1, 16, 32, 16}, {1, 32, 64, 16}},
                      {{1, 16, 32, 16}, {1, 32, 64, 16}},
                      {16, 16, 3, 3});
  MaceRun<GPU, float>(1,
                      {1, 32, 64, 16},
                      {{1, 16, 32, 16}, {1, 32, 64, 16}},
                      {{1, 16, 32, 16}, {1, 32, 64, 16}},
                      {16, 16, 3, 3});
  MaceRun<GPU, half>(2,
                     {1, 32, 64, 16},
                     {{1, 16, 32, 16}, {1, 32, 64, 16}},
                     {{1, 16, 32, 16}, {1, 32, 64, 16}},
                     {16, 16, 3, 3});
}

}  // namespace test
}  // namespace mace
