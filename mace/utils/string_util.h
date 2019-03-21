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

#ifndef MACE_UTILS_STRING_UTIL_H_
#define MACE_UTILS_STRING_UTIL_H_

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

namespace mace {
namespace string_util {

inline void MakeStringInternal(std::stringstream & /*ss*/) {}

template <typename T>
inline void MakeStringInternal(std::stringstream &ss, const T &t) {
  ss << t;
}

template <typename T, typename... Args>
inline void MakeStringInternal(std::stringstream &ss,
                               const T &t,
                               const Args &... args) {
  MakeStringInternal(ss, t);
  MakeStringInternal(ss, args...);
}

class StringFormatter {
 public:
  static std::string Table(const std::string &title,
                           const std::vector<std::string> &header,
                           const std::vector<std::vector<std::string>> &data);
};

}  // namespace string_util

template <typename... Args>
std::string MakeString(const Args &... args) {
  std::stringstream ss;
  string_util::MakeStringInternal(ss, args...);
  return ss.str();
}

template <typename T>
std::string MakeListString(const T *args, size_t size) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < size; ++i) {
    ss << args[i];
    if (i < size - 1) {
      ss << ", ";
    }
  }
  ss << "]";
  return ss.str();
}

template <typename T>
std::string MakeString(const std::vector<T> &args) {
  return MakeListString(args.data(), args.size());
}

// Specializations for already-a-string types.
template <>
inline std::string MakeString(const std::string &str) {
  return str;
}

inline std::string MakeString(const char *c_str) { return std::string(c_str); }

inline std::string ToLower(const std::string &src) {
  std::string dest(src);
  std::transform(src.begin(), src.end(), dest.begin(), ::tolower);
  return dest;
}

inline std::string ToUpper(const std::string &src) {
  std::string dest(src);
  std::transform(src.begin(), src.end(), dest.begin(), ::toupper);
  return dest;
}

std::string ObfuscateString(const std::string &src,
                            const std::string &lookup_table);

std::string ObfuscateString(const std::string &src);

std::string ObfuscateSymbol(const std::string &src);

#ifdef MACE_OBFUSCATE_LITERALS
#define MACE_OBFUSCATE_STRING(str) ObfuscateString(str)
#define MACE_OBFUSCATE_SYMBOL(str) ObfuscateSymbol(str)
#else
#define MACE_OBFUSCATE_STRING(str) (str)
#define MACE_OBFUSCATE_SYMBOL(str) (str)
#endif

std::vector<std::string> Split(const std::string &str, char delims);

}  // namespace mace

#endif  // MACE_UTILS_STRING_UTIL_H_
