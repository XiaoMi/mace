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

#ifndef MACE_UTILS_UTILS_H_
#define MACE_UTILS_UTILS_H_

#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <cstdlib>
#include <utility>
#include <vector>

namespace mace {

// Disable the copy and assignment operator for a class.
#ifndef MACE_DISABLE_COPY_AND_ASSIGN
#define MACE_DISABLE_COPY_AND_ASSIGN(CLASSNAME) \
 private:                                       \
  CLASSNAME(const CLASSNAME &) = delete;        \
  CLASSNAME &operator=(const CLASSNAME &) = delete
#endif

template <typename Integer>
Integer RoundUp(Integer i, Integer factor) {
  return (i + factor - 1) / factor * factor;
}

template <typename Integer, uint32_t factor>
Integer RoundUpDiv(Integer i) {
  return (i + factor - 1) / factor;
}

// Partial specialization of function templates is not allowed
template <typename Integer>
Integer RoundUpDiv4(Integer i) {
  return (i + 3) >> 2;
}

template <typename Integer>
Integer RoundUpDiv8(Integer i) {
  return (i + 7) >> 3;
}

template <typename Integer>
Integer RoundUpDiv(Integer i, Integer factor) {
  return (i + factor - 1) / factor;
}

template <typename Integer>
Integer CeilQuotient(Integer a, Integer b) {
  return (a + b - 1) / b;
}

inline std::string ObfuscateString(const std::string &src,
                                   const std::string &lookup_table) {
  std::string dest;
  dest.resize(src.size());
  for (size_t i = 0; i < src.size(); i++) {
    dest[i] = src[i] ^ lookup_table[i % lookup_table.size()];
  }
  return dest;
}

// ObfuscateString(ObfuscateString(str)) ==> str
inline std::string ObfuscateString(const std::string &src) {
  // Keep consistent with obfuscation in python tools
  return ObfuscateString(src, "Mobile-AI-Compute-Engine");
}

// Obfuscate synbol or path string
inline std::string ObfuscateSymbol(const std::string &src) {
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

#ifdef MACE_OBFUSCATE_LITERALS
#define MACE_OBFUSCATE_STRING(str) ObfuscateString(str)
#define MACE_OBFUSCATE_SYMBOL(str) ObfuscateSymbol(str)
#else
#define MACE_OBFUSCATE_STRING(str) (str)
#define MACE_OBFUSCATE_SYMBOL(str) (str)
#endif

inline std::vector<std::string> Split(const std::string &str, char delims) {
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

inline bool ReadBinaryFile(std::vector<unsigned char> *data,
                           const std::string &filename) {
  std::ifstream ifs(filename, std::ios::in | std::ios::binary);
  if (!ifs.is_open()) {
    return false;
  }
  ifs.seekg(0, ifs.end);
  size_t length = ifs.tellg();
  ifs.seekg(0, ifs.beg);

  data->reserve(length);
  data->insert(data->begin(), std::istreambuf_iterator<char>(ifs),
               std::istreambuf_iterator<char>());
  if (ifs.fail()) {
    return false;
  }
  ifs.close();

  return true;
}

template <typename T>
std::vector<std::string> MapKeys(const std::map<std::string, T> &data) {
  std::vector<std::string> keys;
  for (auto &kv : data) {
    keys.push_back(kv.first);
  }
  return keys;
}

inline bool EnvEnabled(std::string env_name) {
  char *env = getenv(env_name.c_str());
  return !(!env || env[0] == 0 || env[0] == '0');
}

}  // namespace mace
#endif  // MACE_UTILS_UTILS_H_
