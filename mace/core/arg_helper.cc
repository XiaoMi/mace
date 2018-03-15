//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <string>
#include <vector>

#include "mace/core/arg_helper.h"
#include "mace/utils/logging.h"

namespace mace {

ArgumentHelper::ArgumentHelper(const OperatorDef &def) {
  for (auto &arg : def.arg()) {
    if (arg_map_.find(arg.name()) != arg_map_.end()) {
      LOG(WARNING) << "Duplicated argument name found in operator def.";
    }

    arg_map_[arg.name()] = arg;
  }
}

ArgumentHelper::ArgumentHelper(const NetDef &netdef) {
  for (auto &arg : netdef.arg()) {
    MACE_CHECK(arg_map_.count(arg.name()) == 0,
               "Duplicated argument name found in net def.");
    arg_map_[arg.name()] = arg;
  }
}

bool ArgumentHelper::HasArgument(const string &name) const {
  return arg_map_.count(name);
}

namespace {
// Helper function to verify that conversion between types won't loose any
// significant bit.
template <typename InputType, typename TargetType>
bool SupportsLosslessConversion(const InputType &value) {
  return static_cast<InputType>(static_cast<TargetType>(value)) == value;
}
}

#define INSTANTIATE_GET_SINGLE_ARGUMENT(T, fieldname,                         \
                                        enforce_lossless_conversion)          \
  template <>                                                                 \
  T ArgumentHelper::GetSingleArgument<T>(const string &name,                  \
                                         const T &default_value) const {      \
    if (arg_map_.count(name) == 0) {                                          \
      VLOG(3) << "Using default parameter value " << default_value            \
              << " for parameter " << name;                                   \
      return default_value;                                                   \
    }                                                                         \
    MACE_CHECK(arg_map_.at(name).has_##fieldname(), "Argument ", name,        \
               " does not have the right field: expected field " #fieldname); \
    auto value = arg_map_.at(name).fieldname();                               \
    if (enforce_lossless_conversion) {                                        \
      auto supportsConversion =                                               \
          SupportsLosslessConversion<decltype(value), T>(value);              \
      MACE_CHECK(supportsConversion, "Value", value, " of argument ", name,   \
                 "cannot be represented correctly in a target type");         \
    }                                                                         \
    return value;                                                             \
  }                                                                           \
  template <>                                                                 \
  bool ArgumentHelper::HasSingleArgumentOfType<T>(const string &name) const { \
    if (arg_map_.count(name) == 0) {                                          \
      return false;                                                           \
    }                                                                         \
    return arg_map_.at(name).has_##fieldname();                               \
  }

INSTANTIATE_GET_SINGLE_ARGUMENT(float, f, false)
INSTANTIATE_GET_SINGLE_ARGUMENT(double, f, false)
INSTANTIATE_GET_SINGLE_ARGUMENT(bool, i, false)
INSTANTIATE_GET_SINGLE_ARGUMENT(int8_t, i, true)
INSTANTIATE_GET_SINGLE_ARGUMENT(int16_t, i, true)
INSTANTIATE_GET_SINGLE_ARGUMENT(int, i, true)
INSTANTIATE_GET_SINGLE_ARGUMENT(int64_t, i, true)
INSTANTIATE_GET_SINGLE_ARGUMENT(uint8_t, i, true)
INSTANTIATE_GET_SINGLE_ARGUMENT(uint16_t, i, true)
INSTANTIATE_GET_SINGLE_ARGUMENT(size_t, i, true)
INSTANTIATE_GET_SINGLE_ARGUMENT(string, s, false)
#undef INSTANTIATE_GET_SINGLE_ARGUMENT

#define INSTANTIATE_GET_REPEATED_ARGUMENT(T, fieldname,                   \
                                          enforce_lossless_conversion)    \
  template <>                                                             \
  std::vector<T> ArgumentHelper::GetRepeatedArgument<T>(                  \
      const string &name, const std::vector<T> &default_value) const {    \
    if (arg_map_.count(name) == 0) {                                      \
      return default_value;                                               \
    }                                                                     \
    std::vector<T> values;                                                \
    for (const auto &v : arg_map_.at(name).fieldname()) {                 \
      if (enforce_lossless_conversion) {                                  \
        auto supportsConversion =                                         \
            SupportsLosslessConversion<decltype(v), T>(v);                \
        MACE_CHECK(supportsConversion, "Value", v, " of argument ", name, \
                   "cannot be represented correctly in a target type");   \
      }                                                                   \
      values.push_back(v);                                                \
    }                                                                     \
    return values;                                                        \
  }

INSTANTIATE_GET_REPEATED_ARGUMENT(float, floats, false)
INSTANTIATE_GET_REPEATED_ARGUMENT(double, floats, false)
INSTANTIATE_GET_REPEATED_ARGUMENT(bool, ints, false)
INSTANTIATE_GET_REPEATED_ARGUMENT(int8_t, ints, true)
INSTANTIATE_GET_REPEATED_ARGUMENT(int16_t, ints, true)
INSTANTIATE_GET_REPEATED_ARGUMENT(int, ints, true)
INSTANTIATE_GET_REPEATED_ARGUMENT(int64_t, ints, true)
INSTANTIATE_GET_REPEATED_ARGUMENT(uint8_t, ints, true)
INSTANTIATE_GET_REPEATED_ARGUMENT(uint16_t, ints, true)
INSTANTIATE_GET_REPEATED_ARGUMENT(size_t, ints, true)
INSTANTIATE_GET_REPEATED_ARGUMENT(string, strings, false)
#undef INSTANTIATE_GET_REPEATED_ARGUMENT

}  // namespace mace
