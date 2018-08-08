// Copyright 2018 Xiaomi, Inc.  All rights reserved.
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

#ifndef MACE_CORE_REGISTRY_H_
#define MACE_CORE_REGISTRY_H_

#include <functional>
#include <map>
#include <memory>
#include <mutex>  // NOLINT(build/c++11)
#include <string>
#include <vector>

#include "mace/public/mace.h"
#include "mace/utils/logging.h"

namespace mace {

template <class SrcType, class ObjectType, class... Args>
class Registry {
 public:
  typedef std::function<std::unique_ptr<ObjectType>(Args...)> Creator;

  Registry() : registry_() {}

  void Register(const SrcType &key, Creator creator) {
    VLOG(3) << "Registering: " << key;
    std::lock_guard<std::mutex> lock(register_mutex_);
    MACE_CHECK(registry_.count(key) == 0, "Key already registered: ", key);
    registry_[key] = creator;
  }

  std::unique_ptr<ObjectType> Create(const SrcType &key, Args... args) const {
    if (registry_.count(key) == 0) {
      LOG(FATAL) << "Key not registered: " << key;
    }
    return registry_.at(key)(args...);
  }

 private:
  std::map<SrcType, Creator> registry_;
  std::mutex register_mutex_;

  MACE_DISABLE_COPY_AND_ASSIGN(Registry);
};

template <class SrcType, class ObjectType, class... Args>
class Registerer {
 public:
  Registerer(const SrcType &key,
             Registry<SrcType, ObjectType, Args...> *registry,
             typename Registry<SrcType, ObjectType, Args...>::Creator creator) {
    registry->Register(key, creator);
  }

  template <class DerivedType>
  static std::unique_ptr<ObjectType> DefaultCreator(Args... args) {
    return std::unique_ptr<ObjectType>(new DerivedType(args...));
  }
};

#define MACE_CONCATENATE_IMPL(s1, s2) s1##s2
#define MACE_CONCATENATE(s1, s2) MACE_CONCATENATE_IMPL(s1, s2)
#ifdef __COUNTER__
#define MACE_ANONYMOUS_VARIABLE(str) MACE_CONCATENATE(str, __COUNTER__)
#else
#define MACE_ANONYMOUS_VARIABLE(str) MACE_CONCATENATE(str, __LINE__)
#endif

#define MACE_DECLARE_TYPED_REGISTRY(RegistryName, SrcType, ObjectType, ...) \
  typedef Registerer<SrcType, ObjectType, ##__VA_ARGS__>                    \
      Registerer##RegistryName;

#define MACE_DECLARE_REGISTRY(RegistryName, ObjectType, ...)         \
  MACE_DECLARE_TYPED_REGISTRY(RegistryName, std::string, ObjectType, \
                              ##__VA_ARGS__)

#define MACE_REGISTER_TYPED_CLASS(RegistryName, registry, key, ...) \
  Registerer##RegistryName MACE_ANONYMOUS_VARIABLE(RegistryName)( \
      key, registry, Registerer##RegistryName::DefaultCreator<__VA_ARGS__>);

#define MACE_REGISTER_CLASS(RegistryName, registry, key, ...) \
  MACE_REGISTER_TYPED_CLASS(RegistryName, registry, key, __VA_ARGS__)

}  // namespace mace

#endif  // MACE_CORE_REGISTRY_H_
