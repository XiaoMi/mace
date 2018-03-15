//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/utils/logging.h"
#include "mace/utils/utils.h"

namespace mace {

bool GetSourceOrBinaryProgram(const std::string &program_name,
                              const std::string &binary_file_name_prefix,
                              const cl::Context &context,
                              const cl::Device &device,
                              cl::Program *program,
                              bool *is_binary) {
  extern const std::map<std::string, std::vector<unsigned char>>
      kEncryptedProgramMap;
  *is_binary = false;
  auto it_source = kEncryptedProgramMap.find(program_name);
  if (it_source == kEncryptedProgramMap.end()) {
    return false;
  }
  cl::Program::Sources sources;
  std::string content(it_source->second.begin(), it_source->second.end());
  std::string kernel_source = ObfuscateString(content);
  sources.push_back(kernel_source);
  *program = cl::Program(context, sources);

  return true;
}

}  // namespace mace
