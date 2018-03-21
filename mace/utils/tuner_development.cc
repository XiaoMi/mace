//
// Copyright (c) 2017 XiaoMi All rights reserved.
//
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
