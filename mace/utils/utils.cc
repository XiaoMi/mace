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

#include "mace/utils/utils.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "mace/utils/logging.h"

namespace mace {

std::string ObfuscateString(const std::string &src,
                            const std::string &lookup_table) {
  std::string dest;
  dest.resize(src.size());
  for (size_t i = 0; i < src.size(); i++) {
    dest[i] = src[i] ^ lookup_table[i % lookup_table.size()];
  }
  return dest;
}

// ObfuscateString(ObfuscateString(str)) ==> str
std::string ObfuscateString(const std::string &src) {
  // Keep consistent with obfuscation in python tools
  return ObfuscateString(src, "Mobile-AI-Compute-Engine");
}

// Obfuscate synbol or path string
std::string ObfuscateSymbol(const std::string &src) {
  std::string dest = src;
  if (dest.empty()) {
    return dest;
  }
  dest[0] = src[0];  // avoid invalid symbol which starts from 0-9
  const std::string encode_dict =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_";
  for (size_t i = 1; i < src.size(); i++) {
    char ch = src[i];
    int idx;
    if (ch >= '0' && ch <= '9') {
      idx = ch - '0';
    } else if (ch >= 'a' && ch <= 'z') {
      idx = 10 + ch - 'a';
    } else if (ch >= 'A' && ch <= 'Z') {
      idx = 10 + 26 + ch - 'a';
    } else if (ch == '_') {
      idx = 10 + 26 + 26;
    } else {
      dest[i] = ch;
      continue;
    }
    // There is no collision if it's true for every char at every position
    dest[i] = encode_dict[(idx + i + 31) % encode_dict.size()];
  }
  return dest;
}

std::vector<std::string> Split(const std::string &str, char delims) {
  std::vector<std::string> result;
  std::string tmp = str;
  while (!tmp.empty()) {
    size_t next_offset = tmp.find(delims);
    result.push_back(tmp.substr(0, next_offset));
    if (next_offset == std::string::npos) {
      break;
    } else {
      tmp = tmp.substr(next_offset + 1);
    }
  }
  return result;
}

bool ReadBinaryFile(std::vector<unsigned char> *data,
                    const std::string &filename) {
  std::ifstream ifs(filename, std::ios::in | std::ios::binary);
  if (!ifs.is_open()) {
    return false;
  }
  ifs.seekg(0, ifs.end);
  size_t length = ifs.tellg();
  ifs.seekg(0, ifs.beg);

  data->resize(length);
  ifs.read(reinterpret_cast<char *>(data->data()), length);

  if (ifs.fail()) {
    return false;
  }
  ifs.close();

  return true;
}

void MemoryMap(const std::string &file,
               const unsigned char **data,
               size_t *size) {
  int fd = open(file.c_str(), O_RDONLY);
  MACE_CHECK(fd >= 0,
             "Failed to open file ", file, ", error code: ", strerror(errno));
  struct stat st;
  fstat(fd, &st);
  *size = static_cast<size_t>(st.st_size);
  if (*size == 0) {
    return;
  }

  *data = static_cast<const unsigned char *>(
      mmap(nullptr, *size, PROT_READ, MAP_PRIVATE, fd, 0));
  MACE_CHECK(*data != static_cast<const unsigned char *>(MAP_FAILED),
             "Failed to map file ", file, ", error code: ", strerror(errno));

  int ret = close(fd);
  MACE_CHECK(ret == 0,
             "Failed to close file ", file, ", error code: ", strerror(errno));
}

void MemoryUnMap(const unsigned char *data,
                 const size_t &size) {
  if (size == 0) {
    return;
  }
  MACE_CHECK(data != nullptr, "data is null");

  int ret = munmap(const_cast<unsigned char *>(data), size);

  MACE_CHECK(ret == 0,
             "Failed to unmap file, error code: ", strerror(errno));
}

}  // namespace mace
