#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "mace/core/runtime/opencl/cl2_header.h"

namespace mace {

bool GetSourceOrBinaryProgram(const std::string &program_name,
                              const std::string &binary_file_name_prefix,
                              cl::Context &context,
                              cl::Device &device,
                              cl::Program *program,
                              bool *is_binary) {
  extern const std::map<std::string, std::vector<unsigned char>> kCompiledProgramMap;
  *is_binary = true;
  auto it_binary = kCompiledProgramMap.find(binary_file_name_prefix);
  if (it_binary == kCompiledProgramMap.end()) {
    return false;
  }
  *program = cl::Program(context, {device}, {it_binary->second});
  return true;
}

bool GetTuningParams(const char *path,
                     std::unordered_map<std::string, std::vector<unsigned int>> *param_table) {
  extern const std::map<std::string, std::vector<unsigned int>> kTuningParamsData;
  for (auto it = kTuningParamsData.begin(); it != kTuningParamsData.end(); ++it) {
    param_table->emplace(it->first, std::vector<unsigned int>(it->second.begin(), it->second.end()));
  }
  return true;
}

}  // namespace mace
