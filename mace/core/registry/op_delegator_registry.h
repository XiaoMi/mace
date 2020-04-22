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

#ifndef MACE_CORE_REGISTRY_OP_DELEGATOR_REGISTRY_H_
#define MACE_CORE_REGISTRY_OP_DELEGATOR_REGISTRY_H_

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "mace/core/bfloat16.h"
#include "mace/core/ops/op_delegator.h"
#include "mace/core/types.h"
#include "mace/proto/mace.pb.h"
#include "mace/public/mace.h"

namespace mace {
typedef std::function<std::unique_ptr<OpDelegator>(const DelegatorParam &)>
    DelegatorCreator;

struct DelegatorInfo {
  explicit DelegatorInfo(const char *delegator_name,
                         DataType data_type,
                         DeviceType device,
                         ImplType impl_type,
                         const char *tag);
  explicit DelegatorInfo(const char *delegator_name,
                         DataType data_type,
                         DeviceType device,
                         ImplType impl_type);

  std::string ToString() const;

  bool operator==(const DelegatorInfo &info) const;

  std::string delegator_name;
  DataType data_type;
  DeviceType device;
  ImplType impl_type;
  std::string tag;
};

class OpDelegatorRegistry {
 public:
  OpDelegatorRegistry() = default;
  ~OpDelegatorRegistry() = default;

  MaceStatus Register(const DelegatorInfo &key, DelegatorCreator creator);
  DelegatorCreator GetCreator(const DelegatorInfo &key) const;

 private:
  struct HashName {
    size_t operator()(const DelegatorInfo &delegator_info) const {
      return std::hash<std::string>()(delegator_info.ToString());
    }
  };
  std::unordered_map<DelegatorInfo, DelegatorCreator, HashName> registry_;
};

}  // namespace mace

#ifndef MACE_DELEGATOR_KEY_EX_TMP
#define MACE_DELEGATOR_KEY_EX_TMP(delegator_name, device, DT, impl, tag) \
  DelegatorInfo(#delegator_name, DataTypeToEnum<DT>::value, device, impl, #tag)
#endif  // MACE_DELEGATOR_KEY_EX_TMP

#ifndef MACE_DELEGATOR_KEY_EX
#define MACE_DELEGATOR_KEY_EX(delegator_name, device, DT, impl, tag) \
  MACE_DELEGATOR_KEY_EX_TMP(delegator_name, device, DT, impl, tag)
#endif  // MACE_DELEGATOR_KEY_EX

#ifndef MACE_DELEGATOR_KEY
#define MACE_DELEGATOR_KEY(delegator_name, device, DT, impl) \
  DelegatorInfo(#delegator_name, DataTypeToEnum<DT>::value, device, impl)
#endif  // MACE_DELEGATOR_KEY

#ifndef MACE_REGISTER_DELEGATOR
#define MACE_REGISTER_DELEGATOR(registry, class_name, param_name, key)  \
  registry->Register(key, OpDelegator::DefaultCreator<class_name, param_name>)
#endif  // MACE_REGISTER_DELEGATOR

#ifndef MACE_REGISTER_BF16_DELEGATOR
#ifdef MACE_ENABLE_BFLOAT16
#define MACE_REGISTER_BF16_DELEGATOR(registry, class_name, param_name, key) \
  MACE_REGISTER_DELEGATOR(registry, class_name, param_name, key)
#else
#define MACE_REGISTER_BF16_DELEGATOR(registry, class_name, param_name, key)
#endif  // MACE_ENABLE_BFLOAT16
#endif  // MACE_REGISTER_BF16_DELEGATOR

#ifndef MACE_DEFINE_DELEGATOR_CREATOR
#define MACE_DEFINE_DELEGATOR_CREATOR(class_name)            \
  static std::unique_ptr<class_name> Create(                 \
      Workspace *workspace, const DelegatorInfo &key,        \
      const DelegatorParam &param) {                         \
    DelegatorCreator creator =                               \
        workspace->GetDelegatorRegistry()->GetCreator(key);  \
    std::unique_ptr<OpDelegator> delegator = creator(param); \
    return  std::unique_ptr<class_name>(                     \
        static_cast<class_name *>(delegator.release()));     \
  }
#endif  // MACE_DEFINE_DELEGATOR_CREATOR

#endif  // MACE_CORE_REGISTRY_OP_DELEGATOR_REGISTRY_H_
