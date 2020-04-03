// Copyright 2020 The MACE Authors. All Rights Reserved.
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

#ifndef MACE_CORE_OPS_OP_CONDITION_CONTEXT_H_
#define MACE_CORE_OPS_OP_CONDITION_CONTEXT_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "mace/core/types.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/opencl_util.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
class Workspace;
class Device;

// OpConditionContext has all information used for choosing proper Op
class OpConditionContext {
 public:
  typedef std::unordered_map<std::string, std::vector<index_t>> TensorShapeMap;
  OpConditionContext(const Workspace *ws, TensorShapeMap *info);
  ~OpConditionContext() = default;

  void set_operator_def(const OperatorDef *operator_def);

  const OperatorDef *operator_def() const {
    return operator_def_;
  }

  const Workspace *workspace() const {
    return ws_;
  }

  void set_device(Device *device) {
    device_ = device;
  }

  Device *device() const {
    return device_;
  }

  TensorShapeMap *tensor_shape_info() const {
    return tensor_shape_info_;
  }

  void set_output_mem_type(MemoryType type);

  MemoryType output_mem_type() const {
    return output_mem_type_;
  }

  void SetInputInfo(size_t idx, MemoryType mem_type, DataType dt);

  MemoryType GetInputMemType(size_t idx) const;

  DataType GetInputDataType(size_t idx) const;

#ifdef MACE_ENABLE_OPENCL
  void SetInputOpenCLBufferType(size_t idx, OpenCLBufferType buffer_type);
  OpenCLBufferType GetInputOpenCLBufferType(size_t idx) const;
#endif  // MACE_ENABLE_OPENCL

 private:
  const OperatorDef *operator_def_;
  const Workspace *ws_;
  Device *device_;
  TensorShapeMap *tensor_shape_info_;
  // used for memory transform
  std::vector<MemoryType> input_mem_types_;
  std::vector<DataType> input_data_types_;
  MemoryType output_mem_type_;  // there is only one output memory type now.
#ifdef MACE_ENABLE_OPENCL
  std::vector<OpenCLBufferType> input_opencl_buffer_types_;
#endif  // MACE_ENABLE_OPENCL
};
}  // namespace mace

#endif  // MACE_CORE_OPS_OP_CONDITION_CONTEXT_H_
