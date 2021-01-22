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

#ifndef MACE_CORE_RUNTIME_APU_NEURON_DELEGATE_KERNEL_H_
#define MACE_CORE_RUNTIME_APU_NEURON_DELEGATE_KERNEL_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mace/core/runtime/apu/neuron_implementation.h"
#include "mace/core/runtime/apu/neuron_delegate_builder.h"
#include "mace/core/tensor.h"
#include "mace/core/device.h"
#include "mace/core/quantize.h"

namespace mace {
namespace neuron {

// RAII Neuron Model Destructor for use with std::unique_ptr
class NNFreeModel {
 public:
  explicit NNFreeModel(const NeuronApi* neuronapi) : neuronapi_(neuronapi) {}
  void operator()(NeuronModel* model) { neuronapi_->NeuronModel_free(model); }

 private:
  // NeuronApi instance to use. Not owned by this object.
  const NeuronApi* neuronapi_;
};

// RAII Neuron Compilation Destructor for use with std::unique_ptr
class NNFreeCompilation {
 public:
  explicit NNFreeCompilation(const NeuronApi* neuronapi)
      : neuronapi_(neuronapi) {}
  void operator()(NeuronCompilation* compilation) {
    neuronapi_->NeuronCompilation_free(compilation);
  }

 private:
  // NeuronApi instance to use. Not owned by this object.
  const NeuronApi* neuronapi_;
};

// RAII Neuron Execution Destructor for use with std::unique_ptr
class NNFreeExecution {
 public:
  explicit NNFreeExecution(const NeuronApi* neuronapi)
      : neuronapi_(neuronapi) {}
  void operator()(NeuronExecution* execution) {
    neuronapi_->NeuronExecution_free(execution);
  }

 private:
  // NeuronApi instance to use. Not owned by this object.
  const NeuronApi* neuronapi_;
};

// Manage Neuron shared memory handle
class NNMemory {
 public:
  NNMemory(const NeuronApi* neuronapi, size_t size);

  ~NNMemory();

  // NeuronMemory* get_handle() { return nn_memory_handle_; }
  uint8_t* get_data_ptr() { return data_ptr_; }

 private:
  // NeuronApi instance to use. Not owned by this object.
  size_t byte_size_ = 0;
  uint8_t* data_ptr_ = nullptr;
  [[maybe_unused]] NeuronMemory* nn_memory_handle_ = nullptr;
};


// The kernel that represents the node sub set of TF Lite being run on Neuron
// API.
class NeuronDelegateKernel {
 public:
  explicit NeuronDelegateKernel(const NeuronApi* neuronapi, Device *device)
      : neuronapi_(neuronapi),
        nn_model_(nullptr, NNFreeModel(neuronapi_)),
        nn_compilation_(nullptr, NNFreeCompilation(neuronapi_)),
        quantize_util_uint8_(&device->cpu_runtime()->thread_pool()),
        quantize_util_int16_(&device->cpu_runtime()->thread_pool()) {}
  explicit NeuronDelegateKernel(Device *device) :
      NeuronDelegateKernel(NeuronApiImplementation(), device) {}
  ~NeuronDelegateKernel() {
    // Release memory
    for (auto int32_buffer : int32_buffers_) {
      delete [] int32_buffer;
    }
  }
  bool Init(const NetDef* net_def, unsigned const char *model_data, bool load);
  bool Prepare(const char *file_name, bool load, bool store);
  bool Eval(const std::map<std::string, Tensor *> &input_tensors,
            std::map<std::string, Tensor *> *output_tensors);

 private:
  // Access to NNApi.
  const NeuronApi* neuronapi_;
  // Neuron state.
  std::unique_ptr<NeuronModel, NNFreeModel> nn_model_;
  std::unique_ptr<NeuronCompilation, NNFreeCompilation> nn_compilation_;
  std::vector<MaceTensorInfo> input_infos_;
  std::vector<MaceTensorInfo> output_infos_;

  std::unique_ptr<NNMemory> nn_input_memory_;
  std::unique_ptr<NNMemory> nn_output_memory_;

  std::vector<int32_t*> int32_buffers_;

  QuantizeUtil<float, uint8_t> quantize_util_uint8_;
  QuantizeUtil<float, int16_t> quantize_util_int16_;

  size_t total_input_byte_size;
  size_t total_output_byte_size;

  bool SetInputAndOutput(const NetDef* net_def,
                         unsigned const char *model_data);
  bool AddOpsAndTensors(const NetDef* net_def, unsigned const char *model_data);
  bool BuildGraph(const NetDef* net_def, unsigned const char *model_data);
};

}  // namespace neuron
}  // namespace mace

#endif  // MACE_CORE_RUNTIME_APU_NEURON_DELEGATE_KERNEL_H_
