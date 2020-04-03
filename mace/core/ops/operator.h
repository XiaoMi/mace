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

#ifndef MACE_CORE_OPS_OPERATOR_H_
#define MACE_CORE_OPS_OPERATOR_H_

#include <memory>
#include <string>
#include <vector>

#include "mace/core/arg_helper.h"
#include "mace/core/ops/op_construct_context.h"
#include "mace/core/ops/op_context.h"
#include "mace/core/tensor.h"
#include "mace/proto/mace.pb.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/opencl_util.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
class OpInitContext;
// Conventions
// * If there exist format, NHWC is the default format
// * The input/output format of CPU ops with float data type is NCHW
// * The input/output format of GPU ops and CPU Quantization ops is NHWC
// * Inputs' data type is same as the operation data type by default.
// * The outputs' data type is same as the operation data type by default.
class Operation {
 public:
  explicit Operation(OpConstructContext *context);
  virtual ~Operation() = default;

  template<typename T>
  T GetOptionalArg(const std::string &name,
                   const T &default_value) const {
    MACE_CHECK(operator_def_, "operator_def was null!");
    return ProtoArgHelper::GetOptionalArg<OperatorDef, T>(
        *operator_def_, name, default_value);
  }
  template<typename T>
  std::vector<T> GetRepeatedArgs(
      const std::string &name, const std::vector<T> &default_value = {}) const {
    MACE_CHECK(operator_def_, "operator_def was null!");
    return ProtoArgHelper::GetRepeatedArgs<OperatorDef, T>(
        *operator_def_, name, default_value);
  }

  DeviceType device_type() const {
    return static_cast<DeviceType>(operator_def_->device_type());
  }

  const Tensor *Input(unsigned int idx) {
    MACE_CHECK(idx < inputs_.size());
    return inputs_[idx];
  }

  Tensor *Output(int idx) { return outputs_[idx]; }

  int InputSize() { return inputs_.size(); }
  int OutputSize() { return outputs_.size(); }
  const std::vector<const Tensor *> &Inputs() const { return inputs_; }
  const std::vector<Tensor *> &Outputs() { return outputs_; }

  // Run Op asynchronously (depends on device), return a future if not nullptr.
  virtual MaceStatus Init(OpInitContext *);
  virtual MaceStatus Run(OpContext *) = 0;

  const OperatorDef &debug_def() const {
    MACE_CHECK(has_debug_def(), "operator_def was null!");
    return *operator_def_;
  }

  void set_debug_def(
      const std::shared_ptr<OperatorDef> &operator_def) {
    operator_def_ = operator_def;
  }

  bool has_debug_def() const { return operator_def_ != nullptr; }

  inline std::shared_ptr<OperatorDef> operator_def() {
    return operator_def_;
  }

 protected:
  std::shared_ptr<OperatorDef> operator_def_;
  std::vector<const Tensor *> inputs_;
  std::vector<Tensor *> outputs_;

  MACE_DISABLE_COPY_AND_ASSIGN(Operation);
};

// MACE_OP_INPUT_TAGS and MACE_OP_OUTPUT_TAGS are optional features to name the
// indices of the operator's inputs and outputs, in order to avoid confusion.
// For example, for a fully convolution layer that has input, weight and bias,
// you can define its input tags as:
//     MACE_OP_INPUT_TAGS(INPUT, WEIGHT, BIAS);
// And in the code, instead of doing
//     auto& weight = Input(1);
// you can now do
//     auto& weight = Input(WEIGHT);
// to make it more clear.
#define MACE_OP_INPUT_TAGS(first_input, ...) \
  enum _InputTags { first_input = 0, __VA_ARGS__ }
#define MACE_OP_OUTPUT_TAGS(first_input, ...) \
  enum _OutputTags { first_input = 0, __VA_ARGS__ }

}  // namespace mace

#endif  // MACE_CORE_OPS_OPERATOR_H_
