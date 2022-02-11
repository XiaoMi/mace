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

#include "mace/runtimes/apu/v4/neuron_delegate_builder.h"
#include "mace/runtimes/apu/v4/neuron_implementation.h"
#include "mace/runtimes/apu/v4/neuron_delegate_kernel.h"
#include "third_party/apu/android_R2/neuron_types.h"

#ifdef MACE_ENABLE_RPCMEM
#include "mace/runtimes/apu/ion/apu_ion_runtime.h"
#endif  // MACE_ENABLE_RPCMEM

#include <sys/mman.h>
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

NeuronDelegateKernel::NeuronDelegateKernel(const NeuronApi* neuronapi,
                                           Runtime *runtime)
    : neuronapi_(neuronapi),
      nn_model_(nullptr, NNFreeModel(neuronapi_)),
      nn_compilation_(nullptr, NNFreeCompilation(neuronapi_)),
      quantize_util_uint8_(&runtime->thread_pool()),
      quantize_util_int16_(&runtime->thread_pool()) {
#ifdef MACE_ENABLE_RPCMEM
  // Get the rpcmem
  if (runtime->GetRuntimeType() == RT_APU &&
      runtime->GetRuntimeSubType() == RT_SUB_ION) {
    auto apu_ion_runtime = static_cast<ApuIonRuntime *>(runtime);
    rpcmem_ = apu_ion_runtime->GetRpcmem();
    apu_ion_runtime_ = apu_ion_runtime;
  }
#endif
}

bool NeuronDelegateKernel::Init(const NetDef *net_def,
                                unsigned const char *model_data,
                                const char *file_name,
                                const APUPreferenceHint preference_hint,
                                const bool load,
                                const bool store) {
  if (load) {
    if (!file_name || !*file_name) {
      LOG(ERROR) << "Unspecified or empty apu_binary_file=" << file_name;
      return false;
    }
    bool success = SetInputAndOutput(net_def, model_data);
    if (!success) {
      LOG(ERROR) << "SetInputAndOutput failed.";
      return false;
    }
    NeuronModel *restoredModel = nullptr;
    NeuronCompilation *restoredCompilation = nullptr;
    std::ifstream input(file_name, std::ios::binary);
    // copy all data into buffer
    std::vector<unsigned char>
        buffer(std::istreambuf_iterator<char>(input), {});
    int err = neuronapi_->NeuronModel_restoreFromCompiledNetwork(
        &restoredModel, &restoredCompilation, buffer.data(), buffer.size());
    if (err == NEURON_NO_ERROR) {
        LOG(INFO) << "Load pre-compiled model successfully.";
        nn_model_.reset(restoredModel);
        nn_compilation_.reset(restoredCompilation);
        return true;
    }
    // Set ensure_cache_updated=true only for LOAD_OR_STORE policy
    // (i.e. load=true, store=true).
    bool ensure_cache_updated = (load && store) ? true : false;
    if (!ensure_cache_updated) {
      LOG(ERROR) << "Load pre-compiled model failed. err=" << err
                 << " from " << file_name;
      return false;
    }
    // The pre-compiled model is outdated if err == NEURON_BAD_DATA and
    // absent if err == NEURON_UNEXPECTED_NULL
    if ((err != NEURON_BAD_DATA && err != NEURON_UNEXPECTED_NULL)) {
      LOG(ERROR) << "Load pre-compiled model failed, but the cache is"
                    " neither outdated or absent. err=" << err
                 << " from " << file_name;
      return false;
    }
    LOG(INFO) << "Re-compile and update pre-compiled model as the cache is"
                 " outdated or absent. err=" << err << " from " << file_name;
  }

  return CompileModel(net_def,
                      model_data,
                      preference_hint,
                      store,
                      file_name);
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

bool NeuronDelegateKernel::CompileModel(
    const NetDef *net_def,
    unsigned const char *model_data,
    const APUPreferenceHint preference_hint,
    const bool store, const char *file_name) {
  MACE_ASSERT(!nn_model_, "nn_model_ is uninitialized");
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
  NeuronCompilation* compilation = nullptr;
  neuronapi_->NeuronCompilation_create(nn_model_.get(), &compilation);
  if (neuronapi_->NeuronCompilation_setSWDilatedConv != nullptr) {
    neuronapi_->NeuronCompilation_setSWDilatedConv(compilation, true);
  } else {
    LOG(INFO) << "NeuronCompilation_setSWDilatedConv is not supported";
  }
  neuronapi_->NeuronCompilation_setPreference(compilation, preference_hint);
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
    if (!file_name || !*file_name) {
      LOG(ERROR) << "Unspecified or empty apu_store_file=" << file_name;
      return false;
    }
    fp.open(file_name, std::ios::out |
        std::ios :: binary | std::ofstream::trunc);
    fp.write(reinterpret_cast<char*>(buffer), compilationSize);
    fp.close();
    LOG(INFO) << "Store init cache successfully to " << file_name;
    delete[] buffer;
  }
  nn_compilation_.reset(compilation);
  LOG(INFO) << "Neuron compilation success";
  return true;
}

std::vector<NeuronMemory*> NeuronDelegateKernel::PrepareInputTensors(
    const std::map<std::string, Tensor *> &input_tensors,
    NeuronExecution* execution) {
  std::vector<NeuronMemory*> nn_input_memory_buffers;
  size_t input_offset = 0;
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
    // Get the input memory fd
    int input_mem_fd =
        (rpcmem_ == nullptr) ? -1 : rpcmem_->ToFd(tensor->raw_mutable_data());
    if (input_mem_fd > 0) {
      if (tensor->dtype() == input_infos_[i].data_type) {
        // Do Nothing
      } else if (tensor->dtype() == DT_FLOAT &&
          input_infos_[i].data_type == DT_INT16) {
        quantize_util_int16_.QuantizeWithScaleAndZeropoint(
            (const float*)tensor->raw_data(),
            element_size,
            input_infos_[i].scale,
            input_infos_[i].zero_point,
            reinterpret_cast<int16_t*>(input_infos_[i].buf));
        MemInfo mem_info(CPU_BUFFER, DT_INT16, {element_size});
        auto buffer = apu_ion_runtime_->ObtainBuffer(mem_info, RENT_SCRATCH);
        std::memcpy(buffer->mutable_data<void>(), input_infos_[i].buf,
                    byte_size);
        input_mem_fd = rpcmem_->ToFd(buffer->mutable_data<void>());
      } else if (tensor->dtype() == DT_FLOAT &&
          input_infos_[i].data_type == DT_UINT8) {
        quantize_util_uint8_.QuantizeWithScaleAndZeropoint(
            (const float*)tensor->raw_data(),
            element_size,
            input_infos_[i].scale,
            input_infos_[i].zero_point,
            input_infos_[i].buf);
        MemInfo mem_info(CPU_BUFFER, DT_UINT8, {element_size});
        auto buffer = apu_ion_runtime_->ObtainBuffer(mem_info, RENT_SCRATCH);
        std::memcpy(buffer->mutable_data<void>(), input_infos_[i].buf,
                    byte_size);
        input_mem_fd = rpcmem_->ToFd(buffer->mutable_data<void>());
      } else {
        LOG(FATAL) << "Mismatch between input tensor and model input data type";
      }

      NeuronMemory* nn_memory_handle = nullptr;
      neuronapi_->NeuronMemory_createFromFd(byte_size, PROT_READ | PROT_WRITE,
                                            input_mem_fd, 0,
                                            &nn_memory_handle);
      nn_input_memory_buffers.push_back(nn_memory_handle);
      neuronapi_->NeuronExecution_setInputFromMemory(
          execution, i, nullptr, nn_memory_handle, 0, byte_size);
    } else {
      if (tensor->dtype() == input_infos_[i].data_type) {
        std::memcpy(input_infos_[i].buf,
                    (const float*)tensor->raw_data(),
                    byte_size);
      } else if (tensor->dtype() == DT_FLOAT &&
          input_infos_[i].data_type == DT_INT16) {
        quantize_util_int16_.QuantizeWithScaleAndZeropoint(
            (const float*)tensor->raw_data(),
            element_size,
            input_infos_[i].scale,
            input_infos_[i].zero_point,
            reinterpret_cast<int16_t*>(input_infos_[i].buf));
      } else if (tensor->dtype() == DT_FLOAT &&
          input_infos_[i].data_type == DT_UINT8) {
        quantize_util_uint8_.QuantizeWithScaleAndZeropoint(
            (const float*)tensor->raw_data(),
            element_size,
            input_infos_[i].scale,
            input_infos_[i].zero_point,
            input_infos_[i].buf);
      } else {
        LOG(FATAL) << "Mismatch between input tensor and model input data type";
      }
      // Set the input tensor buffers.
      neuronapi_->NeuronExecution_setInput(execution, i, nullptr,
                                           input_infos_[i].buf, byte_size);
    }
  }

  return nn_input_memory_buffers;
}

std::vector<NeuronMemory*>  NeuronDelegateKernel::PrepareOutputTensors(
    std::map<std::string, Tensor *> *output_tensors,
    NeuronExecution* execution) {
  size_t output_offset = 0;
  std::vector<NeuronMemory*> nn_output_memory_buffers;
  for (int i = 0 ; i < static_cast<int>(output_tensors->size()) ; i++) {
    Tensor* tensor = output_tensors->at(output_infos_[i].name);
    // prepare out buffer
    tensor->Resize(output_infos_[i].shape);
    int element_size = output_infos_[i].size;
    int byte_per_element = output_infos_[i].byte_per_element;
    int byte_size = element_size * byte_per_element;
    output_infos_[i].buf = nn_output_memory_->get_data_ptr() + output_offset;
    output_offset += byte_size;
    int output_mem_fd =
        (rpcmem_ == nullptr) ? -1 : rpcmem_->ToFd(tensor->raw_mutable_data());
    NeuronMemory* nn_memory_handle_ = nullptr;
    if (output_mem_fd > 0) {
      neuronapi_->NeuronMemory_createFromFd(byte_size, PROT_READ | PROT_WRITE,
                                            output_mem_fd, 0,
                                            &nn_memory_handle_);
      nn_output_memory_buffers.push_back(nn_memory_handle_);
      neuronapi_->NeuronExecution_setOutputFromMemory(
          execution, i, nullptr, nn_memory_handle_, 0, byte_size);
    } else {
      // Set the output tensor buffers.
      neuronapi_->NeuronExecution_setOutput(execution, i, nullptr,
                                            output_infos_[i].buf, byte_size);
    }
  }

  return nn_output_memory_buffers;
}

void NeuronDelegateKernel::TransposeOutputTensors(
    std::map<std::string, Tensor *> *output_tensors) {
  for (int i = 0 ; i < static_cast<int>(output_tensors->size()) ; i++) {
    Tensor* tensor = output_tensors->at(output_infos_[i].name);
    int element_size = output_infos_[i].size;
    int byte_per_element = output_infos_[i].byte_per_element;
    MACE_ASSERT(element_size == static_cast<int>(tensor->size()),
                "Wrong output size");
    int byte_size = element_size * byte_per_element;

    int output_mem_fd =
        (rpcmem_ == nullptr) ? -1 : rpcmem_->ToFd(tensor->raw_mutable_data());
    if (output_mem_fd > 0) {
      if (tensor->dtype() == output_infos_[i].data_type) {
        // Do nothing
      } else if (tensor->dtype() == DT_FLOAT &&
          output_infos_[i].data_type == DT_INT16) {
        std::memcpy(output_infos_[i].buf, tensor->raw_mutable_data(),
                    byte_size);
        quantize_util_int16_.Dequantize(
            reinterpret_cast<int16_t*>(output_infos_[i].buf),
            element_size,
            output_infos_[i].scale,
            output_infos_[i].zero_point,
            reinterpret_cast<float*>(tensor->raw_mutable_data()));
      } else if (tensor->dtype() == DT_FLOAT &&
          output_infos_[i].data_type == DT_UINT8) {
        std::memcpy(output_infos_[i].buf, tensor->raw_mutable_data(),
                    byte_size);
        quantize_util_uint8_.Dequantize(
            output_infos_[i].buf,
            element_size,
            output_infos_[i].scale,
            output_infos_[i].zero_point,
            reinterpret_cast<float*>(tensor->raw_mutable_data()));
      } else {
        LOG(FATAL) << "Mismatch between output tensor"
                      " and model output data type";
      }
    } else {
      if (tensor->dtype() == output_infos_[i].data_type) {
        std::memcpy(reinterpret_cast<float*>(tensor->raw_mutable_data()),
                    output_infos_[i].buf, byte_size);
      } else if (tensor->dtype() == DT_FLOAT &&
          output_infos_[i].data_type == DT_INT16) {
        quantize_util_int16_.Dequantize(
            reinterpret_cast<int16_t*>(output_infos_[i].buf),
            element_size,
            output_infos_[i].scale,
            output_infos_[i].zero_point,
            reinterpret_cast<float*>(tensor->raw_mutable_data()));
      } else if (tensor->dtype() == DT_FLOAT &&
          output_infos_[i].data_type == DT_UINT8) {
        quantize_util_uint8_.Dequantize(
            output_infos_[i].buf,
            element_size,
            output_infos_[i].scale,
            output_infos_[i].zero_point,
            reinterpret_cast<float*>(tensor->raw_mutable_data()));
      } else {
        LOG(FATAL) << "Mismatch between output tensor"
                      " and model output data type";
      }
    }
  }
}


bool NeuronDelegateKernel::Eval(
    const std::map<std::string, Tensor *> &input_tensors,
    std::map<std::string, Tensor *> *output_tensors,
    const uint8_t boost_hint) {
  NeuronExecution* execution = nullptr;
  neuronapi_->NeuronExecution_create(nn_compilation_.get(), &execution);
  std::unique_ptr<NeuronExecution, NNFreeExecution> execution_unique_ptr(
      execution, NNFreeExecution(neuronapi_));

  MACE_ASSERT(input_tensors.size() == input_infos_.size(), "Wrong inputs num");
  auto nn_input_memory_buffers = PrepareInputTensors(input_tensors, execution);

  MACE_ASSERT(output_tensors->size() == output_infos_.size(),
              "Wrong outputs num");
  auto nn_output_memory_buffers = PrepareOutputTensors(output_tensors,
                                                       execution);

  neuronapi_->NeuronExecution_setBoostHint(execution, boost_hint);
  neuronapi_->NeuronExecution_compute(execution);
  TransposeOutputTensors(output_tensors);

  for (int i = 0; i < static_cast<int>(nn_input_memory_buffers.size()); i++) {
    neuronapi_->NeuronMemory_free(nn_input_memory_buffers[i]);
  }
  for (int i = 0; i < static_cast<int>(nn_output_memory_buffers.size()); i++) {
    neuronapi_->NeuronMemory_free(nn_output_memory_buffers[i]);
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
