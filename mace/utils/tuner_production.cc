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

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace mace {

bool GetTuningParams(
    const char *path,
    std::unordered_map<std::string, std::vector<unsigned int>> *param_table) {
  (void)(path);
  extern const std::map<std::string, std::vector<unsigned int>>
      kTuningParamsData;
  for (auto it = kTuningParamsData.begin(); it != kTuningParamsData.end();
       ++it) {
    param_table->emplace(it->first, std::vector<unsigned int>(
                                        it->second.begin(), it->second.end()));
  }
  return true;
}

}  // namespace mace
