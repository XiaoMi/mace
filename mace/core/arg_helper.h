//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_ARG_HELPER_H_
#define MACE_CORE_ARG_HELPER_H_

#include <map>

#include "mace/core/common.h"
#include "mace/core/public/mace.h"

namespace mace {

using std::string;

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
  static bool HasArgument(const Def &def, const string &name) {
    return ArgumentHelper(def).HasArgument(name);
  }

  template <typename Def, typename T>
  static T GetSingleArgument(const Def &def,
                             const string &name,
                             const T &default_value) {
    return ArgumentHelper(def).GetSingleArgument<T>(name, default_value);
  }

  template <typename Def, typename T>
  static bool HasSingleArgumentOfType(const Def &def, const string &name) {
    return ArgumentHelper(def).HasSingleArgumentOfType<T>(name);
  }

  template <typename Def, typename T>
  static vector<T> GetRepeatedArgument(
      const Def &def,
      const string &name,
      const std::vector<T> &default_value = std::vector<T>()) {
    return ArgumentHelper(def).GetRepeatedArgument<T>(name, default_value);
  }

  explicit ArgumentHelper(const OperatorDef &def);
  explicit ArgumentHelper(const NetDef &netdef);
  bool HasArgument(const string &name) const;

  template <typename T>
  T GetSingleArgument(const string &name, const T &default_value) const;
  template <typename T>
  bool HasSingleArgumentOfType(const string &name) const;
  template <typename T>
  vector<T> GetRepeatedArgument(
      const string &name,
      const std::vector<T> &default_value = std::vector<T>()) const;

 private:
  std::map<string, Argument> arg_map_;
};

}  // namespace mace

#endif  // MACE_CORE_ARG_HELPER_H_
