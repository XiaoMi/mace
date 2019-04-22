// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#include <string>
#include <vector>

#include "mace/core/arg_helper.h"
#include "mace/utils/logging.h"

namespace mace {

ProtoArgHelper::ProtoArgHelper(const OperatorDef &def) {
  for (auto &arg : def.arg()) {
    if (arg_map_.count(arg.name())) {
      LOG(WARNING) << "Duplicated argument " << arg.name()
                   << " found in operator " << def.name();
    }
    arg_map_[arg.name()] = arg;
  }
}

ProtoArgHelper::ProtoArgHelper(const NetDef &netdef) {
  for (auto &arg : netdef.arg()) {
    MACE_CHECK(arg_map_.count(arg.name()) == 0,
               "Duplicated argument found in net def.");
    arg_map_[arg.name()] = arg;
  }
}

namespace {
template <typename InputType, typename TargetType>
inline bool IsCastLossless(const InputType &value) {
  return static_cast<InputType>(static_cast<TargetType>(value)) == value;
}
}

#define MACE_GET_OPTIONAL_ARGUMENT_FUNC(T, fieldname, lossless_conversion)     \
  template <>                                                                  \
  T ProtoArgHelper::GetOptionalArg<T>(const std::string &arg_name,             \
                                      const T &default_value) const {          \
    if (arg_map_.count(arg_name) == 0) {                                       \
      VLOG(3) << "Using default parameter " << default_value << " for "        \
              << arg_name;                                                     \
      return default_value;                                                    \
    }                                                                          \
    MACE_CHECK(arg_map_.at(arg_name).has_##fieldname(), "Argument ", arg_name, \
               " not found!");                                                 \
    auto value = arg_map_.at(arg_name).fieldname();                            \
    if (lossless_conversion) {                                                 \
      const bool castLossless = IsCastLossless<decltype(value), T>(value);     \
      MACE_CHECK(castLossless, "Value", value, " of argument ", arg_name,      \
                 "cannot be casted losslessly to a target type");              \
    }                                                                          \
    return value;                                                              \
  }

MACE_GET_OPTIONAL_ARGUMENT_FUNC(float, f, false)
MACE_GET_OPTIONAL_ARGUMENT_FUNC(bool, i, false)
MACE_GET_OPTIONAL_ARGUMENT_FUNC(int, i, true)
MACE_GET_OPTIONAL_ARGUMENT_FUNC(std::string, s, false)
#undef MACE_GET_OPTIONAL_ARGUMENT_FUNC

#define MACE_GET_REPEATED_ARGUMENT_FUNC(T, fieldname, lossless_conversion) \
  template <>                                                              \
  std::vector<T> ProtoArgHelper::GetRepeatedArgs<T>(                       \
      const std::string &arg_name, const std::vector<T> &default_value)    \
      const {                                                              \
    if (arg_map_.count(arg_name) == 0) {                                   \
      return default_value;                                                \
    }                                                                      \
    std::vector<T> values;                                                 \
    for (const auto &v : arg_map_.at(arg_name).fieldname()) {              \
      if (lossless_conversion) {                                           \
        const bool castLossless = IsCastLossless<decltype(v), T>(v);       \
        MACE_CHECK(castLossless, "Value", v, " of argument ", arg_name,    \
                   "cannot be casted losslessly to a target type");        \
      }                                                                    \
      values.push_back(v);                                                 \
    }                                                                      \
    return values;                                                         \
  }

MACE_GET_REPEATED_ARGUMENT_FUNC(float, floats, false)
MACE_GET_REPEATED_ARGUMENT_FUNC(int, ints, true)
MACE_GET_REPEATED_ARGUMENT_FUNC(int64_t, ints, true)
#undef MACE_GET_REPEATED_ARGUMENT_FUNC

#define MACE_SET_OPTIONAL_ARGUMENT_FUNC(Def, T, fieldname)                     \
  template<>                                                                   \
  void SetProtoArg<T>(Def *def,                                                \
                      const std::string &arg_name,                             \
                      const T &value) {                                        \
    int size = def->arg_size();                                                \
    for (int i = 0; i < size; ++i) {                                           \
      auto arg = def->mutable_arg(i);                                          \
      if (arg->name() == arg_name) {                                           \
        VLOG(3) << "Update old argument value from "                           \
                << arg->fieldname() << " to "                                  \
                << value << " for " << arg_name;                               \
        arg->set_##fieldname(value);                                           \
        return;                                                                \
      }                                                                        \
    }                                                                          \
    VLOG(3) << "Add new argument " << arg_name << "(name: "                    \
            << arg_name << ", value: " << value << ")";                        \
    auto arg = def->add_arg();                                                 \
    arg->set_name(arg_name);                                                   \
    arg->set_##fieldname(value);                                               \
  }

#define MACE_SET_OPTIONAL_ARGUMENT_FUNC_MACRO(Def)     \
  MACE_SET_OPTIONAL_ARGUMENT_FUNC(Def, float, f)       \
  MACE_SET_OPTIONAL_ARGUMENT_FUNC(Def, bool, i)        \
  MACE_SET_OPTIONAL_ARGUMENT_FUNC(Def, int, i)         \
  MACE_SET_OPTIONAL_ARGUMENT_FUNC(Def, int64_t, i)

MACE_SET_OPTIONAL_ARGUMENT_FUNC_MACRO(OperatorDef)
MACE_SET_OPTIONAL_ARGUMENT_FUNC_MACRO(NetDef)
#undef MACE_SET_OPTIONAL_ARGUMENT_FUNC

const std::string OutputMemoryTypeTagName() {
  static const char *kOutputMemTypeArgName = "output_mem_type";
  return kOutputMemTypeArgName;
}

bool IsQuantizedModel(const NetDef &net_def) {
  return
      ProtoArgHelper::GetOptionalArg<NetDef, int>(net_def, "quantize_flag", 0)
          == 1;
}

}  // namespace mace
