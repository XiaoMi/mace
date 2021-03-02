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

#include "mace/core/memory/memory_manager.h"
#include "mace/core/proto/arg_helper.h"
#include "mace/libmace/mace_api_test.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/runtimes/opencl/opencl_runtime.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace test {

namespace {
std::unique_ptr<Tensor> CreateFloatTensor(const std::vector<int64_t> &shape,
                                          MemoryType mem_type) {
  Runtime *runtime = nullptr;
  auto op_context = ops::test::OpTestContext::Get();

#ifdef MACE_ENABLE_OPENCL
  if (mem_type == GPU_BUFFER || mem_type == GPU_IMAGE) {
    runtime = op_context->GetRuntime(RuntimeType::RT_OPENCL);
    auto opencl_runtime = static_cast<OpenclRuntime *>(runtime);
    opencl_runtime->SetUsedMemoryType(mem_type);
  } else {
    runtime = op_context->GetRuntime(RuntimeType::RT_CPU);
  }
#else
  runtime = op_context->GetRuntime(RuntimeType::RT_CPU);
#endif  // MACE_ENABLE_OPENCL

  std::unique_ptr<Tensor> tmp_tensor(new Tensor(
      runtime, DataTypeToEnum<float>::v(), shape));
  Allocator *allocator = runtime->GetMemoryManager(mem_type)->GetAllocator();
  auto mem_shape = runtime->ComputeBufDimFromTensorDim(
      shape, mem_type, BufferContentType::IN_OUT_CHANNEL, 0);
  std::unique_ptr<Buffer> buffer = make_unique<Buffer>(mem_type, DT_FLOAT,
                                                       mem_shape);
  void *memory = nullptr;
  MACE_CHECK_SUCCESS(allocator->New(*buffer, &memory));
  buffer->SetBuf(memory);
  runtime->SetBufferToTensor(std::move(buffer), tmp_tensor.get());

  return tmp_tensor;
}

std::shared_ptr<void> TensorToBuffer(std::unique_ptr<Tensor> tmp_tensor) {
#ifdef MACE_ENABLE_OPENCL
  auto mem_type = tmp_tensor->memory_type();
  if (mem_type == GPU_BUFFER) {
    return std::shared_ptr<cl::Buffer>(
        tmp_tensor->mutable_memory<cl::Buffer>());
  } else if (mem_type == GPU_IMAGE) {
    return std::shared_ptr<cl::Image>(tmp_tensor->mutable_memory<cl::Image>());
  }
#endif  // MACE_ENABLE_OPENCL

  return std::shared_ptr<float>(tmp_tensor->mutable_memory<float>(),
                                std::default_delete<float[]>());
}

}  // namespace

void GenerateInputs(const std::vector<std::string> &input_names,
                    const std::vector<int64_t> &input_shape,
                    std::map<std::string, mace::MaceTensor> *inputs,
                    MemoryType mem_type) {
  for (size_t i = 0; i < input_names.size(); ++i) {
    std::unique_ptr<Tensor> tmp_tensor = CreateFloatTensor(input_shape,
                                                           mem_type);
    // load input
    std::vector<float> input_data;
    ops::test::GenerateRandomRealTypeData(input_shape, &input_data);
    tmp_tensor->CopyBytes(input_data.data(), tmp_tensor->raw_size());
    std::shared_ptr<void> buffer_in = TensorToBuffer(std::move(tmp_tensor));

    (*inputs)[input_names[i]] = mace::MaceTensor(input_shape, buffer_in);
  }
}

void GenerateOutputs(const std::vector<std::string> &output_names,
                     const std::vector<int64_t> &output_shape,
                     std::map<std::string, mace::MaceTensor> *outputs,
                     MemoryType mem_type) {
  MACE_UNUSED(mem_type);
  size_t output_size = output_names.size();
  for (size_t i = 0; i < output_size; ++i) {
    std::unique_ptr<Tensor> tmp_tensor = CreateFloatTensor(output_shape,
                                                           mem_type);
    std::shared_ptr<void> buffer_out = TensorToBuffer(std::move(tmp_tensor));
    (*outputs)[output_names[i]] = mace::MaceTensor(output_shape, buffer_out);
  }
}


class MaceAPITest  : public ::testing::Test {};

namespace {


// The height and width of input and output must be equal.
template <RuntimeType D, typename T>
void MaceRun(const int in_out_size,
             const std::vector<int64_t> &max_shape,
             const std::vector<std::vector<int64_t>> &input_shapes,
             const std::vector<std::vector<int64_t>> &output_shapes,
             const std::vector<int64_t> &filter_shape,
             const MemoryType in_out_mt = CPU_BUFFER) {
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

  MaceEngineConfig config;
#ifdef MACE_ENABLE_OPENCL
  config.SetGPUContext(mace::ops::test::OpTestContext::Get()->gpu_context());
#endif  // MACE_ENABLE_OPENCL
  SetProtoArg(net_def.get(), "runtime_type", static_cast<int>(D));
  auto mem_type = (D == RT_OPENCL ? GPU_IMAGE : CPU_BUFFER);
  SetProtoArg(net_def.get(), "opencl_mem_type", static_cast<int>(mem_type));

  MaceEngine engine(config);
  MaceStatus status = engine.Init(net_def.get(), input_names, output_names,
      reinterpret_cast<unsigned char *>(data.data()), data.size() * sizeof(T));
  EXPECT_EQ(status, MaceStatus::MACE_SUCCESS);

  std::map<std::string, mace::MaceTensor> inputs;
  std::map<std::string, mace::MaceTensor> outputs;

  for (int i = 0; i < 5; ++i) {
    size_t input_shape_size = input_shapes.size();
    for (size_t j = 0; j < input_shape_size; ++j) {
      inputs.clear();
      outputs.clear();
      GenerateInputs(input_names, input_shapes[j], &inputs, in_out_mt);
      GenerateOutputs(output_names, output_shapes[j], &outputs, in_out_mt);
      engine.Run(inputs, &outputs);
    }
  }

  CheckOutputs<D, T>(*net_def, inputs, outputs, data);
}

}  // namespace

TEST_F(MaceAPITest, SingleInputOutput) {
  MaceRun<RT_CPU, float>(1,
                         {1, 32, 32, 16},
                         {{1, 32, 32, 16}},
                         {{1, 32, 32, 16}},
                         {16, 16, 3, 3});
  MaceRun<RT_OPENCL, float>(1,
                            {1, 32, 32, 16},
                            {{1, 32, 32, 16}},
                            {{1, 32, 32, 16}},
                            {16, 16, 3, 3});
  MaceRun<RT_OPENCL, half>(1,
                           {1, 32, 32, 16},
                           {{1, 32, 32, 16}},
                           {{1, 32, 32, 16}},
                           {16, 16, 3, 3});
}

TEST_F(MaceAPITest, MultipleInputOutput) {
  MaceRun<RT_CPU, float>(2,
                         {1, 16, 32, 16},
                         {{1, 16, 32, 16}},
                         {{1, 16, 32, 16}},
                         {16, 16, 3, 3});
  MaceRun<RT_OPENCL, float>(2,
                            {1, 16, 32, 16},
                            {{1, 16, 32, 16}},
                            {{1, 16, 32, 16}},
                            {16, 16, 3, 3});
  MaceRun<RT_OPENCL, half>(2,
                           {1, 16, 32, 16},
                           {{1, 16, 32, 16}},
                           {{1, 16, 32, 16}},
                           {16, 16, 3, 3});
}

TEST_F(MaceAPITest, VariableInputShape) {
  MaceRun<RT_CPU, float>(1,
                         {1, 32, 64, 16},
                         {{1, 16, 32, 16}, {1, 32, 64, 16}},
                         {{1, 16, 32, 16}, {1, 32, 64, 16}},
                         {16, 16, 3, 3});
  MaceRun<RT_OPENCL, float>(1,
                            {1, 32, 64, 16},
                            {{1, 16, 32, 16}, {1, 32, 64, 16}},
                            {{1, 16, 32, 16}, {1, 32, 64, 16}},
                            {16, 16, 3, 3});
  MaceRun<RT_OPENCL, half>(2,
                           {1, 32, 64, 16},
                           {{1, 16, 32, 16}, {1, 32, 64, 16}},
                           {{1, 16, 32, 16}, {1, 32, 64, 16}},
                           {16, 16, 3, 3});
}

}  // namespace test
}  // namespace mace
