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

#ifndef MACE_CORE_OPS_OP_DELEGATOR_H_
#define MACE_CORE_OPS_OP_DELEGATOR_H_

#include <memory>

#include "mace/utils/macros.h"
#include "mace/utils/memory.h"

namespace mace {

enum ImplType {
  REF = 0,
  NEON,
};

#ifdef MACE_ENABLE_NEON
const ImplType kCpuImplType = ImplType::NEON;
#else
const ImplType kCpuImplType = ImplType::REF;
#endif

struct DelegatorParam {
 public:
  DelegatorParam() = default;
  virtual ~DelegatorParam() = default;
};

class OpDelegator {
 public:
  explicit OpDelegator(const DelegatorParam &param) {
    MACE_UNUSED(param);
  }
  virtual ~OpDelegator() = default;

  template<class DerivedType, class ParamType>
  static std::unique_ptr<OpDelegator> DefaultCreator(
      const DelegatorParam &param) {
    return make_unique<DerivedType>(static_cast<const ParamType &>(param));
  }
};

}  // namespace mace

#endif  // MACE_CORE_OPS_OP_DELEGATOR_H_
