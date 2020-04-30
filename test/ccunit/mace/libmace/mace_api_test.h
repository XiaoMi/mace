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

#ifndef MACE_LIBMACE_MACE_API_TEST_H_
#define MACE_LIBMACE_MACE_API_TEST_H_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mace/ops/common/conv_pool_2d_util.h"
#include "mace/ops/ops_test_util.h"
#include "mace/public/mace.h"

namespace mace {
namespace test {

void GenerateInputs(const std::vector<std::string> &input_names,
                           const std::vector<int64_t> &input_shape,
                           std::map<std::string, mace::MaceTensor> *inputs,
                           MemoryType mem_type = CPU_BUFFER);

void GenerateOutputs(const std::vector<std::string> &output_names,
                            const std::vector<int64_t> &output_shape,
                            std::map<std::string, mace::MaceTensor> *outputs,
                            MemoryType mem_type = CPU_BUFFER);

template <typename T>
void Conv3x3(const std::string &input_name,
             const std::string &filter_name,
             const std::string &output_name,
             const std::vector<index_t> &output_shape,
             NetDef *net_def) {
  OperatorDef operator_def;
  ops::test::OpDefBuilder("Conv2D", "Conv2dOp")
      .Input(input_name)
      .Input(filter_name)
      .Output(output_name)
      .AddIntsArg("strides", {1, 1})
      .AddIntArg("padding", Padding::SAME)
      .AddIntsArg("dilations", {1, 1})
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .AddIntArg("data_format", static_cast<int>(DataFormat::AUTO))
      .Finalize(&operator_def);

  OutputShape *shape = operator_def.add_output_shape();
  for (auto dim : output_shape) {
    shape->add_dims(dim);
  }

  net_def->add_op()->CopyFrom(operator_def);
}

template <typename T>
void Relu(const std::string &input_name,
          const std::string &output_name,
          const RuntimeType device_type,
          NetDef *net_def) {
  OperatorDef operator_def;
  ops::test::OpDefBuilder("Activation", "ReluTest")
      .Input(input_name)
      .Output(output_name)
      .AddStringArg("activation", "RELU")
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .AddIntArg("device", static_cast<int>(device_type))
      .AddIntArg("data_format", static_cast<int>(DataFormat::AUTO))
      .Finalize(&operator_def);

  net_def->add_op()->CopyFrom(operator_def);
}

template <typename T>
void AddTensor(const std::string &name,
               const std::vector<int64_t> &shape,
               const int offset,
               const int data_size,
               NetDef *net_def) {
  ConstTensor *tensor_ptr = net_def->add_tensors();
  tensor_ptr->set_name(name);
  tensor_ptr->mutable_dims()->Reserve(shape.size());
  for (auto dim : shape) {
    tensor_ptr->add_dims(dim);
  }
  tensor_ptr->set_offset(offset);
  tensor_ptr->set_data_size(data_size);
  tensor_ptr->set_data_type(DataTypeToEnum<T>::value);
}

template <RuntimeType D, typename T>
void CheckOutputs(const NetDef &net_def,
                  const std::map<std::string, mace::MaceTensor> &inputs,
                  const std::map<std::string, mace::MaceTensor> &outputs,
                  const std::vector<T> &tensor_data) {
  ops::test::OpsTestNet net;
  for (auto input : inputs) {
    auto input_shape = input.second.shape();
    const int64_t data_size = std::accumulate(input_shape.begin(),
                                              input_shape.end(), 1,
                                              std::multiplies<int64_t>());
    std::vector<float> input_data(data_size);
    memcpy(input_data.data(), input.second.data().get(),
           data_size * sizeof(float));
    if (D == RuntimeType::RT_CPU) {
      std::string input_name = input.first + "NHWC";
      net.AddInputFromArray<D, float>(input_name, input_shape, input_data);
      net.TransformDataFormat<D, float>(
          input_name, DataFormat::NHWC, input.first, DataFormat::NCHW);
    } else {
      net.AddInputFromArray<D, float>(input.first, input_shape, input_data);
    }
  }
  auto tensors = net_def.tensors();
  for (auto tensor : tensors) {
    std::vector<index_t> shape = {tensor.dims().begin(), tensor.dims().end()};
    const int64_t data_size = std::accumulate(shape.begin(),
                                              shape.end(), 1,
                                              std::multiplies<int64_t>());
    std::vector<T> data(data_size);
    memcpy(data.data(),
           reinterpret_cast<const T *>(tensor_data.data()) + tensor.offset(),
           tensor.data_size() * sizeof(T));
    net.AddInputFromArray<D, T>(tensor.name(), shape, data, true);
  }
  net.RunNet(net_def, D);

  auto *cpu_runtime =
      ops::test::OpTestContext::Get()->GetRuntime(RuntimeType::RT_CPU);
  for (auto output : outputs) {
    auto &output_shape = output.second.shape();
    const int64_t data_size = std::accumulate(output_shape.begin(),
                                              output_shape.end(), 1,
                                              std::multiplies<float>());
    std::unique_ptr<Tensor> tmp_tensor(new Tensor(
        cpu_runtime, DataTypeToEnum<float>::v(), output.second.shape()));
    cpu_runtime->AllocateBufferForTensor(tmp_tensor.get(),
                                         BufRentType::RENT_SCRATCH);

    float *data = tmp_tensor->mutable_data<float>();
    memcpy(data, output.second.data().get(), data_size * sizeof(float));

    std::string output_name = output.first;
    if (D == RuntimeType::RT_CPU) {
      output_name = output.first + "NHWC";
      net.TransformDataFormat<RuntimeType::RT_CPU, float>(
          output.first, DataFormat::NCHW, output_name, DataFormat::NHWC);
    }
    ops::test::ExpectTensorNear<float>(*tmp_tensor,
                                       *net.GetOutput(output_name.data()),
                                       1e-5);
  }
}
}  // namespace test
}  // namespace mace
#endif  // MACE_LIBMACE_MACE_API_TEST_H_
