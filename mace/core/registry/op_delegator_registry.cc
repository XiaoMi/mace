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
#include <sstream>

#include "mace/utils/logging.h"

namespace mace {

namespace {
const char *kDefaultTag = "general";
}

DelegatorInfo::DelegatorInfo(const char *in_name, DataType in_data_type,
                             DeviceType in_device, ImplType in_impl_type,
                             const char *in_tag)
    : delegator_name(in_name), data_type(in_data_type),
      device(in_device), impl_type(in_impl_type), tag(in_tag) {}

DelegatorInfo::DelegatorInfo(const char *in_name, DataType in_data_type,
                             DeviceType in_device, ImplType in_impl_type)
    : DelegatorInfo(in_name, in_data_type,
                    in_device, in_impl_type, kDefaultTag) {}

std::string DelegatorInfo::ToString() const {
  std::stringstream ss;
  ss << delegator_name << "_" << data_type << "_"
     << device << "_" << impl_type << "_" << tag;
  return ss.str();
}

bool DelegatorInfo::operator==(const DelegatorInfo &info) const {
  return device == info.device && impl_type == info.impl_type &&
      data_type == info.data_type &&
      delegator_name == info.delegator_name && tag == info.tag;
}

MaceStatus OpDelegatorRegistry::Register(const DelegatorInfo &key,
                                         DelegatorCreator creator) {
  MACE_CHECK(registry_.count(key) == 0,
             "Register an exist key: ", key.ToString());
  registry_[key] = std::move(creator);
  return MaceStatus::MACE_SUCCESS;
}

DelegatorCreator OpDelegatorRegistry::GetCreator(
    const DelegatorInfo &key) const {
  if (registry_.count(key) > 0) {
    return registry_.at(key);
  }

  DelegatorInfo info = key;
  if (key.impl_type == ImplType::NEON) {
    if (info.tag != kDefaultTag) {
      info.tag = kDefaultTag;
      if (registry_.count(info) > 0) {
        VLOG(1) << key.ToString()
                << " delegator fall back to " << info.ToString();
        return registry_.at(info);
      }
      info.tag = key.tag;
    }

    info.impl_type = ImplType::REF;
    if (registry_.count(info) > 0) {
      VLOG(1) << key.ToString()
              << " delegator fall back to " << info.ToString();
      return registry_.at(info);
    }
  }

  // for REF
  if (info.tag != kDefaultTag) {
    info.tag = kDefaultTag;
    if (registry_.count(info) > 0) {
      VLOG(1) << key.ToString()
              << " delegator fall back to " << info.ToString();
      return registry_.at(info);
    }
  }

  LOG(FATAL) << "Delegator not exist: " << key.ToString();
  return DelegatorCreator();
}

}  // namespace mace
