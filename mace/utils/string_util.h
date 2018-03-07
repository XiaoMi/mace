//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_UTILS_STRING_UTIL_H_
#define MACE_UTILS_STRING_UTIL_H_

#include <sstream>
#include <string>
#include <vector>

namespace mace {
namespace {

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

}  // namespace

template <typename... Args>
std::string MakeString(const Args &... args) {
  std::stringstream ss;
  MakeStringInternal(ss, args...);
  return ss.str();
}

template <typename T>
std::string MakeString(const std::vector<T> &args) {
  std::stringstream ss;
  ss << "[";
  const size_t size = args.size();
  for (int i = 0; i < size; ++i) {
    ss << args[i];
    if (i < size - 1) {
      ss << ", ";
    }
  }
  ss << "]";
  return ss.str();
}

// Specializations for already-a-string types.
template <>
inline std::string MakeString(const std::string &str) {
  return str;
}

inline std::string MakeString(const char *c_str) { return std::string(c_str); }

}  // namespace mace

#endif  // MACE_UTILS_STRING_UTIL_H_
