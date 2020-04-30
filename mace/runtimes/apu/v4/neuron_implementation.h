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

#ifndef MACE_RUNTIMES_APU_V4_NEURON_IMPLEMENTATION_H_
#define MACE_RUNTIMES_APU_V4_NEURON_IMPLEMENTATION_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "third_party/apu/android_R2/neuron_types.h"

struct NeuronApi {
  bool neuron_exists;
  int32_t neuron_sdk_version;

  // Neuron adapter api function types

  int (*Neuron_getVersion)(uint32_t* version);

  // Creates a shared memory object from a file descriptor.
  // The shared memory is backed by a file descriptor via mmap.
  int (*NeuronMemory_createFromFd)(size_t size, int protect, int fd,
                                   size_t offset, NeuronMemory** memory);

  // Delete a memory object.
  void (*NeuronMemory_free)(NeuronMemory* memory);

  // Create an empty NeuronModel. The model should be constructed with calls to
  // NeuronModel_addOperation and NeuronModel_addOperand.
  int (*NeuronModel_create)(NeuronModel** model);

  // Destroy a model. The model need not have been finished by a call to
  // NeuronModel_free.
  void (*NeuronModel_free)(NeuronModel* model);

  // Indicate that we have finished modifying a model.
  // Required before calling NeuronCompilation_compile.
  int (*NeuronModel_finish)(NeuronModel* model);

  // Gets the supported operations in a model.
  // This function must be called after calling NeuronModel_finish
  int (*NeuronModel_getSupportedOperations)(NeuronModel* model, bool* supported,
                                            uint32_t operationCount);

  // Add an operand to a model. The order in which the operands are added is
  // important. The first one added to a model will have the index value 0, the
  // second 1, etc. These indexes are used as operand identifiers in
  // NeuronModel_addOperation.
  int (*NeuronModel_addOperand)(NeuronModel* model,
                                const NeuronOperandType* type);

  // Sets an operand to a constant value.
  // For scalar values, the content of buffer is copied into the model.
  // For tensor values, a pointer to the buffer is stored within the model.
  int (*NeuronModel_setOperandValue)(NeuronModel* model, int32_t index,
                                     const void* buffer, size_t length);

  // Sets an operand's per channel quantization parameters
  // Sets parameters required by a tensor of type
  // NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL This function must be called for
  // every tensor of type NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL before calling
  // NeuronModel_finish
  int (*NeuronModel_setOperandSymmPerChannelQuantParams)(
      NeuronModel* model, int32_t index,
      const NeuronSymmPerChannelQuantParams* channelQuant);

  // Add an operation to a model.
  // The operands specified by inputs and outputs must have been previously
  // added by calls to NeuronModel_addOperand.
  int (*NeuronModel_addOperation)(NeuronModel* model, NeuronOperationType type,
                                  uint32_t inputCount, const uint32_t* inputs,
                                  uint32_t outputCount,
                                  const uint32_t* outputs);

  // Specfifies which operands will be the model's inputs and outputs.
  // An operand cannot be used for both input and output. Doing so will return
  // an error.
  int (*NeuronModel_identifyInputsAndOutputs)(NeuronModel* model,
                                              uint32_t inputCount,
                                              const uint32_t* inputs,
                                              uint32_t outputCount,
                                              const uint32_t* outputs);

  // Specifies whether NEURON_TENSOR_FLOAT32 is allowed to be calculated with
  // range and/or precision as low as that of the IEEE 754 16-bit floating-point
  // format. By default, NEURON_TENSOR_FLOAT32 must be calculated using at least
  // the range and precision of the IEEE 754 32-bit floating-point format.
  int (*NeuronModel_relaxComputationFloat32toFloat16)(NeuronModel* model,
                                                      bool allow);

  // Create a NeuronCompilation to compile the given model.
  int (*NeuronCompilation_create)(NeuronModel* model,
                                  NeuronCompilation** compilation);

  // Destroy a compilation.
  void (*NeuronCompilation_free)(NeuronCompilation* compilation);

  // Compilation is finished once NeuronCompilation_finish is invoked.
  int (*NeuronCompilation_finish)(NeuronCompilation* compilation);

  // Provides optional caching information for faster re-compilation..
  int (*NeuronCompilation_setCaching)(NeuronCompilation* compilation,
                                      const char* cacheDir,
                                      const uint8_t* token);

  // Create a new execution instance by calling the NeuronExecution_create
  // function.
  int (*NeuronExecution_create)(NeuronCompilation* compilation,
                                NeuronExecution** execution);

  // Destroy an execution.
  void (*NeuronExecution_free)(NeuronExecution* execution);

  // Associate a user buffer with an input of the model of the NeuronExecution.
  int (*NeuronExecution_setInput)(NeuronExecution* execution, int32_t index,
                                  const NeuronOperandType* type,
                                  const void* buffer, size_t length);

  // Associate a user buffer with an output of the model of the NeuronExecution.
  int (*NeuronExecution_setOutput)(NeuronExecution* execution, int32_t index,
                                   const NeuronOperandType* type, void* buffer,
                                   size_t length);

  // Associate a user buffer with an input of the model of the NeuronExecution.
  int (*NeuronExecution_setInputFromMemory)(NeuronExecution* execution,
                                            uint32_t index,
                                            const NeuronOperandType* type,
                                            const NeuronMemory* memory,
                                            size_t offset, size_t length);

  // Associate a user buffer with an output of the model of the NeuronExecution.
  int (*NeuronExecution_setOutputFromMemory)(NeuronExecution* execution,
                                             uint32_t index,
                                             const NeuronOperandType* type,
                                             const NeuronMemory* memory,
                                             size_t offset, size_t length);

  // Schedule synchronous evaluation of the execution.
  // Returns once the execution has completed and the outputs are ready to be
  // consumed.
  int (*NeuronExecution_compute)(NeuronExecution* execution);

  // Create a shared memory region
  int (*ASharedMemory_create)(const char* name, size_t size);

  // Get the compiled network size of the compilation.
  int (*NeuronCompilation_getCompiledNetworkSize)(
      NeuronCompilation* compilation, size_t* size);

  // Store the compiled network.
  int (*NeuronCompilation_storeCompiledNetwork)(NeuronCompilation* compilation,
                                                void* buffer,
                                                const size_t size);

  // Restore the compiled network using user provided buffer.
  int (*NeuronModel_restoreFromCompiledNetwork)(
      NeuronModel** model, NeuronCompilation** compilation,
      const void* buffer, const size_t size);

  // Support dilated convolution.
  int (*NeuronCompilation_setSWDilatedConv)(NeuronCompilation* compilation,
                                            bool allow);
};

/**
 * Load the Neuron implementation from the shared libraries.
 * The NnApi structure is filled with all the pointers. If one function doesn't
 * exist, a null pointer is stored.
 */
const NeuronApi* NeuronApiImplementation();

#endif  // MACE_RUNTIMES_APU_V4_NEURON_IMPLEMENTATION_H_
