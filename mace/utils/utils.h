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

#ifndef MACE_UTILS_UTILS_H_
#define MACE_UTILS_UTILS_H_

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace mace {

// Disable the copy and assignment operator for a class.
#ifndef MACE_DISABLE_COPY_AND_ASSIGN
#define MACE_DISABLE_COPY_AND_ASSIGN(CLASSNAME) \
 private:                                       \
  CLASSNAME(const CLASSNAME &) = delete;        \
  CLASSNAME &operator=(const CLASSNAME &) = delete
#endif

#ifndef MACE_EMPTY_VIRTUAL_DESTRUCTOR
#define MACE_EMPTY_VIRTUAL_DESTRUCTOR(CLASSNAME) \
 public:                                         \
  virtual ~CLASSNAME() {}
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

std::string ObfuscateString(const std::string &src,
                            const std::string &lookup_table);
template <typename Integer>
inline Integer Clamp(Integer in, Integer low, Integer high) {
  return std::max<Integer>(low, std::min<Integer>(in, high));
}

template <typename T>
inline T ScalarSigmoid(T in) {
  if (in > static_cast<T>(0)) {
    return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-in));
  } else {
    T x = std::exp(in);
    return x / (x + static_cast<T>(1));
  }
}

template <typename T>
inline T ScalarTanh(T in) {
  if (in > static_cast<T>(0)) {
    T inv_expa = std::exp(-in);
    return -static_cast<T>(1) +
        static_cast<T>(2) / (static_cast<T>(1) + inv_expa * inv_expa);
  } else {
    T x = std::exp(in);
    return x / (x + static_cast<T>(1));
  }
}

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

bool ReadBinaryFile(std::vector<unsigned char> *data,
                    const std::string &filename);

void MemoryMap(const std::string &file,
               const unsigned char **data,
               size_t *size);

void MemoryUnMap(const unsigned char *data,
                 const size_t &size);

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

template <typename SrcType, typename DstType>
std::vector<DstType> TransposeShape(const std::vector<SrcType> &shape,
                                    const std::vector<int> &dst_dims) {
  size_t shape_dims = shape.size();
  std::vector<DstType> output_shape(shape_dims);
  for (size_t i = 0; i < shape_dims; ++i) {
    output_shape[i] = static_cast<DstType>(shape[dst_dims[i]]);
  }
  return output_shape;
}

}  // namespace mace
#endif  // MACE_UTILS_UTILS_H_
