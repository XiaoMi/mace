//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_UTILS_UTILS_H_
#define MACE_UTILS_UTILS_H_

#include <sys/time.h>
#include <sstream>

namespace mace {
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
Integer CeilQuotient(Integer a, Integer b) {
  return (a + b - 1) / b;
}

inline int64_t NowInMicroSec() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return static_cast<int64_t>(tv.tv_sec * 1000000 + tv.tv_usec);
}

template <typename T>
inline std::string ToString(T v) {
  std::ostringstream ss;
  ss << v;
  return ss.str();
}

inline std::string ObfuscateString(const std::string &src,
                                   const std::string &lookup_table) {
  std::string dest;
  dest.resize(src.size());
  for (int i = 0; i < src.size(); i++) {
    dest[i] = src[i] ^ lookup_table[i % lookup_table.size()];
  }
  return std::move(dest);
}

// ObfuscateString(ObfuscateString(str)) ==> str
inline std::string ObfuscateString(const std::string &src) {
  // Keep consistent with obfuscation in python tools
  return ObfuscateString(src, "Xiaomi-AI-Platform-Mace");
}

// Obfuscate synbol or path string
inline std::string ObfuscateSymbolWithCollision(const std::string &src) {
  std::string dest = ObfuscateString(src);
  const std::string encode_dict =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
  for (int i = 0; i < src.size(); i++) {
    dest[i] = encode_dict[dest[i] % encode_dict.size()];
  }
  return std::move(dest);
}

#ifdef MACE_OBFUSCATE_LITERALS
#define MACE_OBFUSCATE_STRING(str) ObfuscateString(str)
// This table is delibratedly selected to avoid '\0' in genereated literal
#define MACE_OBFUSCATE_SYMBOLS(str) ObfuscateString(str, "!@#$%^&*()+?")
// OpenCL will report error if there is name collision
#define MACE_KERNRL_NAME(name) ObfuscateSymbolWithCollision(name)
#else
#define MACE_OBFUSCATE_STRING(str) (str)
#define MACE_OBFUSCATE_SYMBOLS(str) (str)
#define MACE_KERNRL_NAME(name) (name)
#endif

}  //  namespace mace
#endif  //  MACE_UTILS_UTILS_H_
