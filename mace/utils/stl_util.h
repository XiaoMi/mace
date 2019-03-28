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

#ifndef MACE_UTILS_STL_UTIL_H_
#define MACE_UTILS_STL_UTIL_H_

#include <algorithm>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace mace {

template <typename T>
std::vector<std::string> MapKeys(const std::map<std::string, T> &data) {
  std::vector<std::string> keys;
  for (auto &kv : data) {
    keys.push_back(kv.first);
  }
  return keys;
}

}  // namespace mace

#endif  // MACE_UTILS_STL_UTIL_H_
