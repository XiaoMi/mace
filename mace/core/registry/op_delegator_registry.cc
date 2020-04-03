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

#include "mace/core/registry/op_delegator_registry.h"

#include <utility>

#include "mace/utils/logging.h"

namespace mace {

MaceStatus OpDelegatorRegistry::Register(const std::string &key,
                                         DelegatorCreator creator) {
  MACE_CHECK(registry_.count(key) == 0, "Register an exist key.");
  registry_[key] = std::move(creator);
  return MaceStatus::MACE_SUCCESS;
}

DelegatorCreator OpDelegatorRegistry::GetCreator(const std::string &key) const {
  MACE_CHECK(registry_.count(key) > 0, key, " not exist.");
  return registry_.at(key);
}

template<> const char *DType<float>::name_ = "float";
template<> const char *DType<int>::name_ = "int";
template<> const char *DType<uint8_t>::name_ = "uint8_t";

}  // namespace mace
