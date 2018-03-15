//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/utils/logging.h"

namespace mace {

bool GetSourceOrBinaryProgram(const std::string &program_name,
                              const std::string &binary_file_name_prefix,
                              const cl::Context &context,
                              const cl::Device &device,
                              cl::Program *program,
                              bool *is_binary) {
  extern const std::map<std::string, std::vector<unsigned char>>
      kCompiledProgramMap;
  *is_binary = true;
  auto it_binary = kCompiledProgramMap.find(binary_file_name_prefix);
  if (it_binary == kCompiledProgramMap.end()) {
    return false;
  }
  *program = cl::Program(context, {device}, {it_binary->second});
  return true;
}

}  // namespace mace
