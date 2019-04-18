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

#ifdef MACE_ENABLE_OPENCL

#include <thread>  // NOLINT(build/c++11)

#include "mace/libmace/mace_api_test.h"

namespace mace {
namespace test {

class MaceMTAPITest  : public ::testing::Test {};

namespace {

// The height and width of input and output must be equal.
void MaceRunFunc(const int in_out_size) {
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  for (int i = 0; i < in_out_size; ++i) {
    input_names.push_back(MakeString("input", i));
    output_names.push_back(MakeString("output", i));
  }
  std::string filter_tensor_name = "filter";

  const std::vector<std::vector<int64_t>> input_shapes = {{1, 32, 32, 16}};
  const std::vector<std::vector<int64_t>> output_shapes = {{1, 32, 32, 16}};
  const std::vector<int64_t> filter_shape = {16, 16, 3, 3};

  std::shared_ptr<NetDef> net_def(new NetDef());

  std::vector<half> data;
  ops::test::GenerateRandomRealTypeData<half>(filter_shape, &data);
  AddTensor<half>(
      filter_tensor_name, filter_shape, 0, data.size(), net_def.get());

  for (size_t i = 0; i < input_names.size(); ++i) {
    InputOutputInfo *info = net_def->add_input_info();
    info->set_data_format(static_cast<int>(DataFormat::NHWC));
    info->set_name(input_names[i]);
    for (auto d : input_shapes[0]) {
      info->add_dims(static_cast<int>(d));
    }
  }
  for (size_t i = 0; i < output_names.size(); ++i) {
    InputOutputInfo *info = net_def->add_output_info();
    info->set_name(output_names[i]);
  }
  for (size_t i = 0; i < output_names.size(); ++i) {
    Conv3x3<half>(input_names[i], filter_tensor_name,
                  output_names[i], output_shapes[0],
                  net_def.get());
  }

  MaceEngineConfig config(DeviceType::GPU);
  config.SetGPUContext(mace::ops::test::OpTestContext::Get()->gpu_context());

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

  CheckOutputs<DeviceType::GPU, half>(*net_def, inputs, outputs, data);
}

}  // namespace

TEST_F(MaceMTAPITest, MultipleThread) {
  const int thread_num = 10;
  std::vector<std::thread> threads;
  for (int i = 0; i < thread_num; ++i) {
    threads.push_back(std::thread(MaceRunFunc, 1));
  }
  for (auto &t : threads) {
    t.join();
  }
}

}  // namespace test
}  // namespace mace

#endif  // MACE_ENABLE_OPENCL
