//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_OPERATOR_H
#define MACE_CORE_OPERATOR_H

#include "mace/core/arg_helper.h"
#include "mace/core/future.h"
#include "mace/public/mace.h"
#include "mace/core/registry.h"
#include "mace/core/tensor.h"
#include "mace/core/workspace.h"

namespace mace {

class OperatorBase {
 public:
  explicit OperatorBase(const OperatorDef &operator_def, Workspace *ws);
  virtual ~OperatorBase() noexcept {}

  inline bool HasArgument(const string &name) const {
    MACE_CHECK(operator_def_, "operator_def was null!");
    return ArgumentHelper::HasArgument(*operator_def_, name);
  }
  template <typename T>
  inline T GetSingleArgument(const string &name, const T &default_value) const {
    MACE_CHECK(operator_def_, "operator_def was null!");
    return ArgumentHelper::GetSingleArgument<OperatorDef, T>(
        *operator_def_, name, default_value);
  }
  template <typename T>
  inline bool HasSingleArgumentOfType(const string &name) const {
    MACE_CHECK(operator_def_, "operator_def was null!");
    return ArgumentHelper::HasSingleArgumentOfType<OperatorDef, T>(
        *operator_def_, name);
  }
  template <typename T>
  inline std::vector<T> GetRepeatedArgument(
      const string &name, const std::vector<T> &default_value = {}) const {
    MACE_CHECK(operator_def_, "operator_def was null!");
    return ArgumentHelper::GetRepeatedArgument<OperatorDef, T>(
        *operator_def_, name, default_value);
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
  virtual bool Run(StatsFuture *future) = 0;

  inline const OperatorDef &debug_def() const {
    MACE_CHECK(has_debug_def(), "operator_def was null!");
    return *operator_def_;
  }

  inline void set_debug_def(
      const std::shared_ptr<const OperatorDef> &operator_def) {
    operator_def_ = operator_def;
  }

  inline bool has_debug_def() const { return operator_def_ != nullptr; }

 protected:
  Workspace *operator_ws_;
  std::shared_ptr<const OperatorDef> operator_def_;
  std::vector<const Tensor *> inputs_;
  std::vector<Tensor *> outputs_;

  DISABLE_COPY_AND_ASSIGN(OperatorBase);
};

template <DeviceType D, class T>
class Operator : public OperatorBase {
 public:
  explicit Operator(const OperatorDef &operator_def, Workspace *ws)
      : OperatorBase(operator_def, ws) {
    for (const string &input_str : operator_def.input()) {
      const Tensor *tensor = ws->GetTensor(input_str);
      MACE_CHECK(tensor != nullptr, "op ", operator_def.type(),
                 ": Encountered a non-existing input tensor: ", input_str);
      inputs_.push_back(tensor);
    }

    for (const string &output_str : operator_def.output()) {
      if (ws->HasTensor(output_str)) {
        outputs_.push_back(ws->GetTensor(output_str));
      } else {
        outputs_.push_back(MACE_CHECK_NOTNULL(ws->CreateTensor(
            output_str, GetDeviceAllocator(D), DataTypeToEnum<T>::v())));
      }
    }
  }
  virtual bool Run(StatsFuture *future) override = 0;
  ~Operator() noexcept override {}
};

// OP_INPUT_TAGS and OP_OUTPUT_TAGS are optional features to name the indices of
// the
// operator's inputs and outputs, in order to avoid confusion. For example, for
// a fully convolution layer that has input, weight and bias, you can define its
// input tags as:
//     OP_INPUT_TAGS(INPUT, WEIGHT, BIAS);
// And in the code, instead of doing
//     auto& weight = Input(1);
// you can now do
//     auto& weight = Input(WEIGHT);
// to make it more clear.
#define OP_INPUT_TAGS(first_input, ...) \
  enum _InputTags { first_input = 0, __VA_ARGS__ }
#define OP_OUTPUT_TAGS(first_input, ...) \
  enum _OutputTags { first_input = 0, __VA_ARGS__ }

class OpKeyBuilder {
 public:
  explicit OpKeyBuilder(const char *op_name);

  OpKeyBuilder &Device(DeviceType device);

  OpKeyBuilder &TypeConstraint(const char *attr_name, const DataType allowed);

  template <typename T>
  OpKeyBuilder &TypeConstraint(const char *attr_name);

  const std::string Build();

 private:
  std::string op_name_;
  DeviceType device_type_;
  std::map<std::string, DataType> type_constraint_;
};

template <typename T>
OpKeyBuilder &OpKeyBuilder::TypeConstraint(const char *attr_name) {
  return this->TypeConstraint(attr_name, DataTypeToEnum<T>::value);
}

class OperatorRegistry {
 public:
  typedef Registry<std::string, OperatorBase, const OperatorDef &, Workspace *>
    RegistryType;
  OperatorRegistry();
  ~OperatorRegistry() = default;
  RegistryType *registry() { return &registry_; };
  std::unique_ptr<OperatorBase> CreateOperator(const OperatorDef &operator_def,
                                               Workspace *ws,
                                               DeviceType type,
                                               const NetMode mode) const;

 private:
  RegistryType registry_;
  DISABLE_COPY_AND_ASSIGN(OperatorRegistry);
};

MACE_DECLARE_REGISTRY(OpRegistry,
                      OperatorBase,
                      const OperatorDef &,
                      Workspace *);

#define REGISTER_OPERATOR(op_registry, name, ...) \
  MACE_REGISTER_CLASS(OpRegistry, op_registry->registry(), name, __VA_ARGS__)

}  //  namespace mace

#endif  // MACE_CORE_OPERATOR_H
