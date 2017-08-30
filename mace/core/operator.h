//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_OPERATOR_H
#define MACE_CORE_OPERATOR_H

#include "mace/core/proto_utils.h"
#include "mace/core/common.h"
#include "mace/proto/mace.pb.h"
#include "mace/core/tensor.h"
#include "mace/core/registry.h"
#include "mace/core/workspace.h"

namespace mace {

class OperatorBase {
 public:
  explicit OperatorBase(const OperatorDef &operator_def, Workspace *ws);
  virtual ~OperatorBase() noexcept {}

  inline bool HasArgument(const string &name) const {
    REQUIRE(operator_def_, "operator_def was null!");
    return ArgumentHelper::HasArgument(*operator_def_, name);
  }
  template<typename T>
  inline T GetSingleArgument(const string &name, const T &default_value) const {
    REQUIRE(operator_def_, "operator_def was null!");
    return ArgumentHelper::GetSingleArgument<OperatorDef, T>(
        *operator_def_, name, default_value);
  }
  template<typename T>
  inline bool HasSingleArgumentOfType(const string &name) const {
    REQUIRE(operator_def_, "operator_def was null!");
    return ArgumentHelper::HasSingleArgumentOfType<OperatorDef, T>(
        *operator_def_, name);
  }
  template<typename T>
  inline vector<T> GetRepeatedArgument(
      const string &name,
      const vector<T> &default_value = {}) const {
    REQUIRE(operator_def_, "operator_def was null!");
    return ArgumentHelper::GetRepeatedArgument<OperatorDef, T>(
        *operator_def_, name, default_value);
  }

  inline const Tensor *Input(int idx) {
    CHECK(idx < inputs_.size());
    return inputs_[idx];
  }

  inline Tensor *Output(int idx) {
    return outputs_[idx];
  }

  inline int InputSize() { return inputs_.size(); }
  inline int OutputSize() { return outputs_.size(); }
  inline const vector<const Tensor *> &Inputs() const { return inputs_; }
  inline const vector<Tensor *> &Outputs() { return outputs_; }

  virtual bool Run() = 0;

  inline const OperatorDef &debug_def() const {
    REQUIRE(has_debug_def(), "operator_def was null!");
    return *operator_def_;
  }

  inline void set_debug_def(
      const std::shared_ptr<const OperatorDef> &operator_def) {
    operator_def_ = operator_def;
  }

  inline bool has_debug_def() const {
    return operator_def_ != nullptr;
  }

 protected:
  Workspace *operator_ws_;
  std::shared_ptr<const OperatorDef> operator_def_;
  vector<const Tensor *> inputs_;
  vector<Tensor *> outputs_;

 DISABLE_COPY_AND_ASSIGN(OperatorBase);
};

template <DeviceType D, class T>
class Operator : public OperatorBase {
 public:
  explicit Operator(const OperatorDef &operator_def, Workspace *ws)
      : OperatorBase(operator_def, ws) {
    for (const string &input_str : operator_def.input()) {
      const Tensor *tensor = ws->GetTensor(input_str);
      REQUIRE(
          tensor != nullptr,
          "op ",
          operator_def.type(),
          ": Encountered a non-existing input tensor: ",
          input_str);
      inputs_.push_back(tensor);
    }

    for (const string &output_str : operator_def.output()) {
      outputs_.push_back(CHECK_NOTNULL(ws->CreateTensor(output_str,
                         DeviceContext<D>::allocator(),
                         DataTypeToEnum<T>::v())));
    }
  }
  virtual bool Run() = 0;
  ~Operator() noexcept override {}
};

typedef Registry<std::string, OperatorBase, const OperatorDef &, Workspace *>
    OperatorRegistry;
typedef Registry<std::string, OperatorBase, const OperatorDef &, Workspace *> *(
    *RegistryFunction)();
std::map<int32_t, OperatorRegistry *> *gDeviceTypeRegistry();

struct DeviceTypeRegisterer {
  explicit DeviceTypeRegisterer(int32_t type, RegistryFunction func) {
    if (gDeviceTypeRegistry()->count(type)) {
      LOG(ERROR) << "Device type " << type
                 << "registered twice. This should not happen. Did you have "
                     "duplicated numbers assigned to different devices?";
      std::exit(1);
    }
    // Calling the registry function to get the actual registry pointer.
    gDeviceTypeRegistry()->emplace(type, func());
  }
};

#define MACE_REGISTER_DEVICE_TYPE(type, registry_function) \
  namespace {                                               \
  static DeviceTypeRegisterer MACE_ANONYMOUS_VARIABLE(     \
      DeviceType)(type, &registry_function);                \
  }

MACE_DECLARE_REGISTRY(
    CPUOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);

#define REGISTER_CPU_OPERATOR_CREATOR(key, ...) \
  MACE_REGISTER_CREATOR(CPUOperatorRegistry, key, __VA_ARGS__)
#define REGISTER_CPU_OPERATOR(name, ...)                           \
  MACE_REGISTER_CLASS(CPUOperatorRegistry, name, __VA_ARGS__)

unique_ptr<OperatorBase> CreateOperator(
    const OperatorDef &operator_def,
    Workspace *ws,
    DeviceType type);

} //  namespace mace

#endif //MACE_CORE_OPERATOR_H
