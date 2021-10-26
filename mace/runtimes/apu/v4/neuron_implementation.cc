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

#include "mace/runtimes/apu/v4/neuron_implementation.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>

#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif  // __ANDROID__

#define NEURONAPI_LOG(format, ...) fprintf(stderr, format "\n", __VA_ARGS__);

void* LoadFunction(void* handle, const char* name, bool optional) {
  if (handle == nullptr) {
    return nullptr;
  }
  void* fn = dlsym(handle, name);
  if (fn == nullptr && !optional) {
    NEURONAPI_LOG("nnapi error: unable to open function %s", name);
  }
  return fn;
}

#ifndef __ANDROID__
// Add /dev/shm implementation of shared memory for non-Android platforms
int ASharedMemory_create(const char* name, size_t size) {
  int fd = shm_open(name, O_RDWR | O_CREAT, 0644);
  if (fd < 0) {
    return fd;
  }
  int result = ftruncate(fd, size);
  if (result < 0) {
    close(fd);
    return -1;
  }
  return fd;
}
#endif  // __ANDROID__

#define LOAD_FUNCTION(handle, name, neuronapi_obj)  \
  neuronapi_obj.name = reinterpret_cast<name##_fn>( \
      LoadFunction(handle, #name, /*optional*/ false));

#define LOAD_FUNCTION_OPTIONAL(handle, name, neuronapi_obj) \
  neuronapi_obj.name = reinterpret_cast<name##_fn>(         \
      LoadFunction(handle, #name, /*optional*/ true));

#define LOAD_FUNCTION_RENAME(handle, name, symbol, neuronapi_obj) \
  neuronapi_obj.name = reinterpret_cast<name##_fn>(               \
      LoadFunction(handle, symbol, /*optional*/ false));

const NeuronApi LoadNeuronApi() {
  NeuronApi neuron_api = {};
  neuron_api.neuron_sdk_version = 0;

  void* libneuron_adapter = nullptr;
  libneuron_adapter =
      dlopen("libneuronusdk_adapter.mtk.so", RTLD_LAZY | RTLD_LOCAL);
  if (libneuron_adapter == nullptr) {
    libneuron_adapter = dlopen("libneuron_adapter_mgvi.so", RTLD_LAZY | RTLD_LOCAL);
  }
  if (libneuron_adapter == nullptr) {
    libneuron_adapter = dlopen("libneuron_adapter.so", RTLD_LAZY | RTLD_LOCAL);
  }
  if (libneuron_adapter == nullptr) {
    NEURONAPI_LOG("NeuronApi error: unable to open library %s",
                  "libneuronusdk_adapter.mtk.so, "
                  "libneuron_adapter_mgvi.so, "
                  "libneuron_adapter.so");
  }

  neuron_api.neuron_exists = libneuron_adapter != nullptr;

  LOAD_FUNCTION(libneuron_adapter, Neuron_getVersion, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronMemory_createFromFd, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronMemory_free, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronModel_create, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronModel_free, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronModel_finish, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronModel_getSupportedOperations,
                neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronModel_addOperand, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronModel_setOperandValue, neuron_api);
  LOAD_FUNCTION(libneuron_adapter,
                NeuronModel_setOperandSymmPerChannelQuantParams, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronModel_addOperation, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronModel_identifyInputsAndOutputs,
                neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronModel_relaxComputationFloat32toFloat16,
                neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronExecution_create, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronCompilation_create, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronCompilation_free, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronCompilation_finish, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronCompilation_setCaching, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronCompilation_setPreference, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronExecution_free, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronExecution_setInput, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronExecution_setOutput, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronExecution_setInputFromMemory,
                neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronExecution_setOutputFromMemory,
                neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronExecution_compute, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronExecution_setBoostHint, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronCompilation_getCompiledNetworkSize,
                neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronCompilation_storeCompiledNetwork,
                neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronModel_restoreFromCompiledNetwork,
                neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronCompilation_setSWDilatedConv,
                neuron_api);

  // ASharedMemory_create has different implementations in Android depending on
  // the partition. Generally it can be loaded from libandroid.so but in vendor
  // partition (e.g. if a HAL wants to use Neuron) it is only accessible through
  // libcutils.
#ifdef __ANDROID__
  void* libandroid = nullptr;
  libandroid = dlopen("libandroid.so", RTLD_LAZY | RTLD_LOCAL);
  if (libandroid != nullptr) {
    LOAD_FUNCTION(libandroid, ASharedMemory_create, neuron_api);
  } else {
    void* cutils_handle = dlopen("libcutils.so", RTLD_LAZY | RTLD_LOCAL);
    if (cutils_handle != nullptr) {
      LOAD_FUNCTION_RENAME(cutils_handle, ASharedMemory_create,
                           "ashmem_create_region", neuron_api);
    } else {
      NEURONAPI_LOG("neuron error: unable to open neither libraries %s and %s",
                    "libandroid.so", "libcutils.so");
    }
  }
#else
  // Mock ASharedMemory_create only if libneuralnetworks.so was successfully
  // loaded. This ensures identical behaviour on platforms which use this
  // implementation, but don't have libneuralnetworks.so library, and
  // platforms which use nnapi_implementation_disabled.cc stub.
  if (libneuron_adapter != nullptr) {
    neuron_api.ASharedMemory_create = ASharedMemory_create;
  }
#endif  // __ANDROID__

  return neuron_api;
}

const NeuronApi* NeuronApiImplementation() {
  static const NeuronApi neuron_api = LoadNeuronApi();
  return &neuron_api;
}
