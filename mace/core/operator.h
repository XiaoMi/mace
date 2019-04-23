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

#ifndef MACE_CORE_OPERATOR_H_
#define MACE_CORE_OPERATOR_H_

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "mace/core/arg_helper.h"
#include "mace/core/op_context.h"
#include "mace/core/tensor.h"
#include "mace/core/workspace.h"
#include "mace/proto/mace.pb.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/opencl_util.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {

// OpConditionContext has all information used for choosing proper Op
class OpConditionContext {
 public:
  typedef std::unordered_map<std::string, std::vector<index_t>> TensorShapeMap;
  OpConditionContext(const Workspace *ws, TensorShapeMap *info);
  ~OpConditionContext() = default;

  void set_operator_def(const OperatorDef* operator_def);

  inline const OperatorDef *operator_def() const {
    return operator_def_;
  }

  inline const Workspace *workspace() const {
    return ws_;
  }

  inline void set_device(Device* device) {
    device_ = device;
  }

  inline Device *device() const {
    return device_;
  }

  inline TensorShapeMap *tensor_shape_info() const {
    return tensor_shape_info_;
  }

  void set_output_mem_type(MemoryType type);

  inline MemoryType output_mem_type() const {
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

// memory_optimizer, device
class OpConstructContext {
  typedef std::unordered_map<std::string, std::vector<index_t>> TensorShapeMap;

 public:
  explicit OpConstructContext(Workspace *ws);
  ~OpConstructContext() = default;

  void set_operator_def(std::shared_ptr<OperatorDef> operator_def);

  inline std::shared_ptr<OperatorDef> operator_def() const {
    return operator_def_;
  }

  inline Workspace *workspace() const {
    return ws_;
  }

  inline void set_device(Device* device) {
    device_ = device;
  }

  inline Device *device() const {
    return device_;
  }
#ifdef MACE_ENABLE_OPENCL
  inline MemoryType GetOpMemoryType() const {
    return static_cast<MemoryType>(
        ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
            *operator_def_, OutputMemoryTypeTagName(),
            static_cast<int>(MemoryType::CPU_BUFFER)));
  }
#endif  // MACE_ENABLE_OPENCL

 private:
  std::shared_ptr<OperatorDef> operator_def_;
  Workspace *ws_;
  Device *device_;
};

// memory_optimizer, device
class OpInitContext {
 public:
  explicit OpInitContext(Workspace *ws, Device *device = nullptr);
  ~OpInitContext() = default;

  inline Workspace *workspace() const {
    return ws_;
  }

  inline void set_device(Device *device) {
    device_ = device;
  }

  inline Device *device() const {
    return device_;
  }

 private:
  Workspace *ws_;
  Device *device_;
};

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

  template <typename T>
  inline T GetOptionalArg(const std::string &name,
                          const T &default_value) const {
    MACE_CHECK(operator_def_, "operator_def was null!");
    return ProtoArgHelper::GetOptionalArg<OperatorDef, T>(
        *operator_def_, name, default_value);
  }
  template <typename T>
  inline std::vector<T> GetRepeatedArgs(
      const std::string &name, const std::vector<T> &default_value = {}) const {
    MACE_CHECK(operator_def_, "operator_def was null!");
    return ProtoArgHelper::GetRepeatedArgs<OperatorDef, T>(
        *operator_def_, name, default_value);
  }

  inline DeviceType device_type() const {
    return static_cast<DeviceType>(operator_def_->device_type());
  }

  inline const Tensor *Input(unsigned int idx) {
    MACE_CHECK(idx < inputs_.size());
    return inputs_[idx];
  }

  inline Tensor *Output(int idx) { return outputs_[idx]; }

  inline int InputSize() { return inputs_.size(); }
  inline int OutputSize() { return outputs_.size(); }
  inline const std::vector<const Tensor *> &Inputs() const { return inputs_; }
  inline const std::vector<Tensor *> &Outputs() { return outputs_; }

  // Run Op asynchronously (depends on device), return a future if not nullptr.
  virtual MaceStatus Init(OpInitContext *);
  virtual MaceStatus Run(OpContext *) = 0;

  inline const OperatorDef &debug_def() const {
    MACE_CHECK(has_debug_def(), "operator_def was null!");
    return *operator_def_;
  }

  inline void set_debug_def(
      const std::shared_ptr<OperatorDef> &operator_def) {
    operator_def_ = operator_def;
  }

  inline bool has_debug_def() const { return operator_def_ != nullptr; }

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


struct OpRegistrationInfo {
 public:
  typedef std::function<std::unique_ptr<Operation>(OpConstructContext *)>
      OpCreator;
  typedef std::function<std::set<DeviceType>(OpConditionContext *)>
      DevicePlacer;
  typedef std::function<void(OpConditionContext *)> MemoryTypeSetter;
  typedef std::function<std::vector<DataFormat>(OpConditionContext *)>
      DataFormatSelector;

  OpRegistrationInfo();

  void AddDevice(DeviceType);

  void Register(const std::string &key, OpCreator creator);

  std::set<DeviceType> devices;
  std::unordered_map<std::string, OpCreator> creators;
  DevicePlacer device_placer;
  MemoryTypeSetter memory_type_setter;
  DataFormatSelector data_format_selector;
};

class OpConditionBuilder {
 public:
  explicit OpConditionBuilder(const std::string &type);

  const std::string type() const;

  OpConditionBuilder &SetDevicePlacerFunc(
      OpRegistrationInfo::DevicePlacer placer);

  // If you set input memory type for specified Op,
  // you must call OpConditionContext::set_output_mem_type
  OpConditionBuilder &SetInputMemoryTypeSetter(
      OpRegistrationInfo::MemoryTypeSetter setter);

  OpConditionBuilder &SetInputsDataFormatSelector(
      OpRegistrationInfo::DataFormatSelector selector);

  void Finalize(OpRegistrationInfo *info) const;

 private:
  std::string type_;
  OpRegistrationInfo::DevicePlacer placer_;
  OpRegistrationInfo::MemoryTypeSetter memory_type_setter_;
  OpRegistrationInfo::DataFormatSelector data_format_selector_;
};


class OpRegistryBase {
 public:
  OpRegistryBase() = default;
  virtual ~OpRegistryBase() = default;
  MaceStatus Register(const std::string &op_type,
                      const DeviceType device_type,
                      const DataType dt,
                      OpRegistrationInfo::OpCreator creator);

  MaceStatus Register(const OpConditionBuilder &builder);

  const std::set<DeviceType> AvailableDevices(
      const std::string &op_type, OpConditionContext *context) const;

  void GetInOutMemoryTypes(
      const std::string &op_type, OpConditionContext *context) const;

  const std::vector<DataFormat> InputsDataFormat(
      const std::string &op_type, OpConditionContext *context) const;

  std::unique_ptr<Operation> CreateOperation(
      OpConstructContext *context,
      DeviceType device_type) const;

  template <class DerivedType>
  static std::unique_ptr<Operation> DefaultCreator(
      OpConstructContext *context) {
    return std::unique_ptr<Operation>(new DerivedType(context));
  }

 private:
  std::unordered_map<
      std::string,
      std::unique_ptr<OpRegistrationInfo>> registry_;
  MACE_DISABLE_COPY_AND_ASSIGN(OpRegistryBase);
};

#define MACE_REGISTER_OP(op_registry, op_type, class_name, device, dt) \
  op_registry->Register(op_type,                                       \
                        device,                                        \
                        DataTypeToEnum<dt>::value,                     \
                        OpRegistryBase::DefaultCreator<class_name<device, dt>>)

#define MACE_REGISTER_OP_CONDITION(op_registry, builder) \
  op_registry->Register(builder)

}  // namespace mace

#endif  // MACE_CORE_OPERATOR_H_
