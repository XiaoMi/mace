// Copyright 2021 The MACE Authors. All Rights Reserved.
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

#include "mace/core/runtime/apu/neuron_delegate_builder.h"
#include "mace/core/runtime/apu/neuron_implementation.h"
#include "mace/core/runtime/apu/neuron_delegate_kernel.h"
#include "third_party/apu/neuron_types.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

namespace mace {
namespace neuron {

namespace {
}

NNMemory::NNMemory(const NeuronApi* /*neuronapi*/,
                   size_t size) {
  byte_size_ = size;
  data_ptr_ = reinterpret_cast<uint8_t*>(malloc(size));
}

NNMemory::~NNMemory() {
  if (data_ptr_) {
    free(data_ptr_);
  }
}


bool NeuronDelegateKernel::Init(const NetDef *net_def,
                                unsigned const char *model_data, bool load) {
  if (load) {
    bool success = SetInputAndOutput(net_def, model_data);
    if (!success) {
      LOG(ERROR) << "SetInputAndOutput failed.";
      return false;
    }
    return true;
  }
  if (!nn_model_) {
    LOG(INFO) << "Creating Neuron model";
    NeuronModel* model = nullptr;
    // Creating Neuron model
    neuronapi_->NeuronModel_create(&model);
    nn_model_.reset(model);
    bool success = BuildGraph(net_def, model_data);
    if (!success) {
      LOG(ERROR) << "BuildGraph failed.";
      return false;
    }
  }
  return true;
}

bool NeuronDelegateKernel::BuildGraph(
    const NetDef *net_def, unsigned const char *model_data) {
  LOG(INFO) << "BuildGraph.";
  // Build the ops and tensors.
  bool success = AddOpsAndTensors(net_def, model_data);
  if (!success) {
    LOG(ERROR) << "AddOpsAndTensors failed.";
    return false;
  }
  neuronapi_->
      NeuronModel_relaxComputationFloat32toFloat16(nn_model_.get(), true);
  // Create shared memory pool for inputs and outputs.
  nn_input_memory_.reset(
      new NNMemory(neuronapi_, total_input_byte_size));
  nn_output_memory_.reset(
      new NNMemory(neuronapi_, total_output_byte_size));
  return true;
}

bool NeuronDelegateKernel::Prepare(const char *file_name,
                                   bool load, bool store) {
  if (load) {
    NeuronModel *restoredModel = nullptr;
    NeuronCompilation *restoredCompilation = nullptr;
    std::ifstream input(file_name, std::ios::binary);

    // copies all data into buffer
    std::vector<unsigned char>
        buffer(std::istreambuf_iterator<char>(input), {});
    int err = neuronapi_->NeuronModel_restoreFromCompiledNetwork(
        &restoredModel, &restoredCompilation, buffer.data(), buffer.size());
    if (err != NEURON_NO_ERROR) {
      LOG(ERROR) << "Load pre-compiled model failed.";
      return false;
    }
    LOG(INFO) << "Load pre-compiled model successfully.";
    nn_compilation_.reset(restoredCompilation);
    return true;
  }
  NeuronCompilation* compilation = nullptr;
  neuronapi_->NeuronCompilation_create(nn_model_.get(), &compilation);
  neuronapi_->NeuronCompilation_setSWDilatedConv(compilation, true);
  const int compilation_result =
      neuronapi_->NeuronCompilation_finish(compilation);
  if (compilation_result != NEURON_NO_ERROR) {
    neuronapi_->NeuronCompilation_free(compilation);
    compilation = nullptr;
    LOG(ERROR) << "Neuron compilation failed";
    return false;
  }
  if (store) {
    size_t compilationSize;
    int err = neuronapi_->NeuronCompilation_getCompiledNetworkSize(
        compilation, &compilationSize);
    if (err != NEURON_NO_ERROR) {
      LOG(ERROR) << "Store init cache failed";
      return false;
    }
    // Allocate user buffer with the compilation size.
    uint8_t *buffer = new uint8_t[compilationSize];
    // Copy the cache into user allocated buffer.
    neuronapi_->NeuronCompilation_storeCompiledNetwork(
        compilation, buffer, compilationSize);
    std::ofstream fp;
    fp.open(file_name, std::ios::out |
        std::ios :: binary | std::ofstream::trunc);
    fp.write(reinterpret_cast<char*>(buffer), compilationSize);
    fp.close();
    LOG(INFO) << "Store init cache successfully";
    delete[] buffer;
  }
  LOG(INFO) << "Neuron compilation success";
  nn_compilation_.reset(compilation);
  return true;
}


bool NeuronDelegateKernel::Eval(
    const std::map<std::string, Tensor *> &input_tensors,
    std::map<std::string, Tensor *> *output_tensors) {

  NeuronExecution* execution = nullptr;
  neuronapi_->NeuronExecution_create(nn_compilation_.get(), &execution);
  std::unique_ptr<NeuronExecution, NNFreeExecution> execution_unique_ptr(
      execution, NNFreeExecution(neuronapi_));

  MACE_ASSERT(input_tensors.size() == input_infos_.size(), "Wrong inputs num");
  MACE_ASSERT(output_tensors.size() == output_infos.size(),
              "Wrong outputs num");

  size_t input_offset = 0;
  // prepare input
  for (int i = 0 ; i < static_cast<int>(input_tensors.size()) ; i++) {
    Tensor* tensor = input_tensors.at(input_infos_[i].name);
    // check size
    int element_size = input_infos_[i].size;
    int byte_per_element = input_infos_[i].byte_per_element;
    int byte_size = element_size * byte_per_element;
    MACE_ASSERT(element_size == static_cast<int>(tensor->size()),
                "Wrong input size");
    input_infos_[i].buf = nn_input_memory_->get_data_ptr() + input_offset;
    input_offset += byte_size;
    // quantize
    if (input_infos_[i].data_type == DT_INT16) {
      quantize_util_int16_.QuantizeWithScaleAndZeropoint(
          (const float*)tensor->raw_data(),
          element_size,
          input_infos_[i].scale,
          input_infos_[i].zero_point,
          reinterpret_cast<int16_t*>(input_infos_[i].buf));
    } else if (input_infos_[i].data_type == DT_FLOAT) {
      std::memcpy(input_infos_[i].buf,
                    (const float*)tensor->raw_data(),
                    byte_size);
    } else {
      quantize_util_uint8_.QuantizeWithScaleAndZeropoint(
          (const float*)tensor->raw_data(),
          element_size,
          input_infos_[i].scale,
          input_infos_[i].zero_point,
          input_infos_[i].buf);
    }
    // Set the input tensor buffers.
    neuronapi_->NeuronExecution_setInput(execution,
        i, nullptr, input_infos_[i].buf, byte_size);
  }

  // prepare output
  size_t output_offset = 0;
  for (int i = 0 ; i < static_cast<int>(output_tensors->size()) ; i++) {
    int element_size = output_infos_[i].size;
    int byte_per_element = output_infos_[i].byte_per_element;
    int byte_size = element_size * byte_per_element;
    output_infos_[i].buf = nn_output_memory_->get_data_ptr() + output_offset;
    output_offset += byte_size;

    // Set the output tensor buffers.
    neuronapi_->NeuronExecution_setOutput(execution, i,
        nullptr, output_infos_[i].buf, byte_size);
  }

  // running computation
  neuronapi_->NeuronExecution_compute(execution);


  // process output
  for (int i = 0 ; i < static_cast<int>(output_tensors->size()) ; i++) {
    Tensor* tensor = output_tensors->at(output_infos_[i].name);

    // prepare out buffer
    tensor->SetDtype(DT_FLOAT);
    tensor->Resize(output_infos_[i].shape);
    int element_size = output_infos_[i].size;
    int byte_per_element = output_infos_[i].byte_per_element;
    MACE_ASSERT(element_size == static_cast<int>(tensor->size()),
                "Wrong output size");

    // dequantize
    if (output_infos_[i].data_type == DT_INT16) {
      quantize_util_int16_.Dequantize(
          reinterpret_cast<int16_t*>(output_infos_[i].buf),
          element_size,
          output_infos_[i].scale,
          output_infos_[i].zero_point,
          reinterpret_cast<float*>(tensor->raw_mutable_data()));
    } else if (output_infos_[i].data_type == DT_FLOAT) {
        std::memcpy(reinterpret_cast<float*>(tensor->raw_mutable_data()),
                    output_infos_[i].buf,
                    element_size * byte_per_element);
    } else {
      quantize_util_uint8_.Dequantize(
          output_infos_[i].buf,
          element_size,
          output_infos_[i].scale,
          output_infos_[i].zero_point,
          reinterpret_cast<float*>(tensor->raw_mutable_data()));
    }
  }
  return true;
}

bool NeuronDelegateKernel::SetInputAndOutput(const NetDef* net_def,
                                             unsigned const char *model_data) {
  NeuronOpBuilder builder(neuronapi_, net_def, model_data, nn_model_.get());
  bool success = builder.SetInputAndOutputImp();
  if (!success) {
    LOG(ERROR) << "SetInputAndOutput failed.";
    return false;
  }
  total_input_byte_size = builder.GetTotalInputByteSize();
  total_output_byte_size = builder.GetTotalOutputByteSize();
  input_infos_ = builder.GetInputInfos();
  output_infos_ = builder.GetOutputInfos();
  // Create shared memory pool for inputs and outputs.
  nn_input_memory_.reset(
      new NNMemory(neuronapi_, total_input_byte_size));
  nn_output_memory_.reset(
      new NNMemory(neuronapi_, total_output_byte_size));
  return true;
}

bool NeuronDelegateKernel::AddOpsAndTensors(const NetDef* net_def,
                                            unsigned const char *model_data) {
  NeuronOpBuilder builder(neuronapi_, net_def, model_data, nn_model_.get());
  bool success = builder.AddOpsAndTensorsImp(&int32_buffers_);
  if (!success) {
    LOG(ERROR) << "AddOpsAndTensors failed.";
    return false;
  }
  total_input_byte_size = builder.GetTotalInputByteSize();
  total_output_byte_size = builder.GetTotalOutputByteSize();
  input_infos_ = builder.GetInputInfos();
  output_infos_ = builder.GetOutputInfos();
  return true;
}

}  // namespace neuron
}  // namespace mace
