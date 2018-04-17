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

#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace mace {

bool GetTuningParams(
    const char *path,
    std::unordered_map<std::string, std::vector<unsigned int>> *param_table) {
  if (path != nullptr) {
    std::ifstream ifs(path, std::ios::binary | std::ios::in);
    if (ifs.is_open()) {
      int64_t num_params = 0;
      ifs.read(reinterpret_cast<char *>(&num_params), sizeof(num_params));
      while (num_params--) {
        int32_t key_size = 0;
        ifs.read(reinterpret_cast<char *>(&key_size), sizeof(key_size));
        std::string key(key_size, ' ');
        ifs.read(&key[0], key_size);

        int32_t params_size = 0;
        ifs.read(reinterpret_cast<char *>(&params_size), sizeof(params_size));
        int32_t params_count = params_size / sizeof(unsigned int);
        std::vector<unsigned int> params(params_count);
        for (int i = 0; i < params_count; ++i) {
          ifs.read(reinterpret_cast<char *>(&params[i]), sizeof(unsigned int));
        }
        param_table->emplace(key, params);
      }
      ifs.close();
    } else {
      return false;
    }
  }
  return true;
}

}  // namespace mace
