#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "mace/core/runtime/opencl/cl2_header.h"


namespace mace {

namespace {
inline void DecryptOpenCLSource(const std::vector<unsigned char> &src,
                                std::vector<unsigned char> *dst) {
  dst->reserve(src.size());

  // Keep consistent with encrypt in python tool
  const std::string decrypt_lookup_table = "Xiaomi-AI-Platform-Mace";
  size_t lookup_table_size = decrypt_lookup_table.size();
  for (int i = 0; i < src.size(); i++) {
    dst->push_back(src[i] ^ decrypt_lookup_table[i % lookup_table_size]);
  }
}
}  // namespace

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
  std::vector<unsigned char> decrypt_source;
  DecryptOpenCLSource(it_source->second, &decrypt_source);
  sources.push_back(std::string(decrypt_source.begin(), decrypt_source.end()));
  *program = cl::Program(context, sources);

  return true;
}

bool GetTuningParams(const char *path,
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
