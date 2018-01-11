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
inline std::string ObfuscateSymbol(const std::string &src) {
  std::string dest = src;
  if (dest.empty()) {
    return dest;
  }
  dest[0] = src[0]; // avoid invalid symbol which starts from 0-9
  const std::string encode_dict =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_";
  for (int i = 1; i < src.size(); i++) {
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
  return std::move(dest);
}

#ifdef MACE_OBFUSCATE_LITERALS
#define MACE_OBFUSCATE_STRING(str) ObfuscateString(str)
#define MACE_OBFUSCATE_SYMBOL(str) ObfuscateSymbol(str)
#else
#define MACE_OBFUSCATE_STRING(str) (str)
#define MACE_OBFUSCATE_SYMBOL(str) (str)
#endif

}  //  namespace mace
#endif  //  MACE_UTILS_UTILS_H_
