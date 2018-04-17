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

#ifndef MACE_CORE_ARG_HELPER_H_
#define MACE_CORE_ARG_HELPER_H_

#include <string>
#include <vector>
#include <map>

#include "mace/public/mace.h"
#include "mace/public/mace_types.h"

namespace mace {

/**
 * @brief A helper class to index into arguments.
 *
 * This helper helps us to more easily index into a set of arguments
 * that are present in the operator. To save memory, the argument helper
 * does not copy the operator def, so one would need to make sure that the
 * lifetime of the OperatorDef object outlives that of the ArgumentHelper.
 */
class ArgumentHelper {
 public:
  template <typename Def>
  static bool HasArgument(const Def &def, const std::string &name) {
    return ArgumentHelper(def).HasArgument(name);
  }

  template <typename Def, typename T>
  static T GetSingleArgument(const Def &def,
                             const std::string &name,
                             const T &default_value) {
    return ArgumentHelper(def).GetSingleArgument<T>(name, default_value);
  }

  template <typename Def, typename T>
  static bool HasSingleArgumentOfType(const Def &def, const std::string &name) {
    return ArgumentHelper(def).HasSingleArgumentOfType<T>(name);
  }

  template <typename Def, typename T>
  static std::vector<T> GetRepeatedArgument(
      const Def &def,
      const std::string &name,
      const std::vector<T> &default_value = std::vector<T>()) {
    return ArgumentHelper(def).GetRepeatedArgument<T>(name, default_value);
  }

  explicit ArgumentHelper(const OperatorDef &def);
  explicit ArgumentHelper(const NetDef &netdef);
  bool HasArgument(const std::string &name) const;

  template <typename T>
  T GetSingleArgument(const std::string &name, const T &default_value) const;
  template <typename T>
  bool HasSingleArgumentOfType(const std::string &name) const;
  template <typename T>
  std::vector<T> GetRepeatedArgument(
      const std::string &name,
      const std::vector<T> &default_value = std::vector<T>()) const;

 private:
  std::map<std::string, Argument> arg_map_;
};

}  // namespace mace

#endif  // MACE_CORE_ARG_HELPER_H_
