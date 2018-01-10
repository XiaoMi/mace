#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/utils/utils.h"

namespace mace {

bool GetSourceOrBinaryProgram(const std::string &program_name,
                              const std::string &binary_file_name_prefix,
                              cl::Context &context,
                              cl::Device &device,
                              cl::Program *program,
                              bool *is_binary) {
  extern const std::map<std::string, std::vector<unsigned char>> kEncryptedProgramMap;
  *is_binary = false;
  auto it_source = kEncryptedProgramMap.find(program_name);
  if (it_source == kEncryptedProgramMap.end()) {
    return false;
  }
  cl::Program::Sources sources;
  std::string kernel_source(it_source->second.begin(), it_source->second.end());
  sources.push_back(ObfuscateString(kernel_source));
  *program = cl::Program(context, sources);

  return true;
}

}  // namespace mace
