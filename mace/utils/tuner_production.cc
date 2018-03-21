//
// Copyright (c) 2017 XiaoMi All rights reserved.
//
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace mace {

bool GetTuningParams(
    const char *path,
    std::unordered_map<std::string, std::vector<unsigned int>> *param_table) {
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
