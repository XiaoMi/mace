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

#ifndef MACE_CORE_ARG_HELPER_H_
#define MACE_CORE_ARG_HELPER_H_

#include <map>
#include <string>
#include <vector>

#include "mace/proto/mace.pb.h"

namespace mace {

// Refer to caffe2
class ProtoArgHelper {
 public:
  template <typename Def, typename T>
  static T GetOptionalArg(const Def &def,
                          const std::string &arg_name,
                          const T &default_value) {
    return ProtoArgHelper(def).GetOptionalArg<T>(arg_name, default_value);
  }

  template <typename Def, typename T>
  static std::vector<T> GetRepeatedArgs(
      const Def &def,
      const std::string &arg_name,
      const std::vector<T> &default_value = std::vector<T>()) {
    return ProtoArgHelper(def).GetRepeatedArgs<T>(arg_name, default_value);
  }

  explicit ProtoArgHelper(const OperatorDef &def);
  explicit ProtoArgHelper(const NetDef &netdef);

  template <typename T>
  T GetOptionalArg(const std::string &arg_name, const T &default_value) const;
  template <typename T>
  std::vector<T> GetRepeatedArgs(
      const std::string &arg_name,
      const std::vector<T> &default_value = std::vector<T>()) const;

 private:
  std::map<std::string, Argument> arg_map_;
};

template <typename T>
void SetProtoArg(OperatorDef *op_def,
                 const std::string &arg_name,
                 const T&value);

template <typename T>
void SetProtoArg(NetDef *op_def,
                 const std::string &arg_name,
                 const T&value);

const std::string OutputMemoryTypeTagName();

bool IsQuantizedModel(const NetDef &def);

}  // namespace mace

#endif  // MACE_CORE_ARG_HELPER_H_
