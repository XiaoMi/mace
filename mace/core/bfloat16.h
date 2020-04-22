// Copyright 2020 The MACE Authors. All Rights Reserved.
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

#ifndef MACE_CORE_BFLOAT16_H_
#define MACE_CORE_BFLOAT16_H_

#ifdef MACE_ENABLE_BFLOAT16

#include <algorithm>
#include <cmath>
#include <sstream>

namespace mace {

union Sphinx {
  uint32_t i;
  float f;

  Sphinx(uint32_t value) : i(value) {}

  Sphinx(float value) : f(value) {}
};

class BFloat16 {
 public:
  BFloat16() : data_(0) {}

  // we need implicit transformation, so `explicit` keyword is not used
  BFloat16(float value) : data_(Sphinx(value).i >> 16) {}  // NOLINT

  operator float() const {
    return Sphinx(static_cast<uint32_t>(data_ << 16)).f;
  }

  operator double() const {
    return static_cast<double>(
        Sphinx(static_cast<uint32_t>(data_ << 16)).f);
  }

  operator int() const {
    return static_cast<int>(Sphinx(static_cast<uint32_t>(data_ << 16)).f);
  }

  template<typename T>
  void operator=(T value) {
    data_ = Sphinx(static_cast<float>(value)).i >> 16;
  }

  BFloat16 operator-() const {
    return BFloat16(-(Sphinx(static_cast<uint32_t>(data_ << 16)).f));
  }

  template<typename T>
  BFloat16 operator+(T value) const {
    return BFloat16(Sphinx(
        static_cast<uint32_t>(data_ << 16)).f + static_cast<float>(value));
  }

  template<typename T>
  BFloat16 operator-(T value) const {
    return BFloat16(Sphinx(
        static_cast<uint32_t>(data_ << 16)).f - static_cast<float>(value));
  }

  template<typename T>
  BFloat16 operator*(T value) const {
    return BFloat16(Sphinx(
        static_cast<uint32_t>(data_ << 16)).f * static_cast<float>(value));
  }

  template<typename T>
  BFloat16 operator/(T value) const {
    return BFloat16(Sphinx(
        static_cast<uint32_t>(data_ << 16)).f / static_cast<float>(value));
  }

  template<typename T>
  bool operator>(T value) const {
    return Sphinx(
        static_cast<uint32_t>(data_ << 16)).f > static_cast<float>(value);
  }

  template<typename T>
  bool operator>=(T value) const {
    return Sphinx(
        static_cast<uint32_t>(data_ << 16)).f >= static_cast<float>(value);
  }

  template<typename T>
  bool operator<(T value) const {
    return Sphinx(
        static_cast<uint32_t>(data_ << 16)).f < static_cast<float>(value);
  }

  template<typename T>
  bool operator<=(T value) const {
    return Sphinx(
        static_cast<uint32_t>(data_ << 16)).f <= static_cast<float>(value);
  }

  template<typename T>
  bool operator==(T value) const {
    return Sphinx(
        static_cast<uint32_t>(data_ << 16)).f == static_cast<float>(value);
  }

  template<typename T>
  void operator+=(T value) {
    data_ = Sphinx(Sphinx(static_cast<uint32_t>(data_ << 16)).f +
        static_cast<float>(value)).i >> 16;
  }

  template<typename T>
  void operator/=(T value) {
    data_ = Sphinx(Sphinx(static_cast<uint32_t>(data_ << 16)).f /
        static_cast<float>(value)).i >> 16;
  }

  template<typename T>
  void operator-=(T value) {
    data_ = Sphinx(Sphinx(static_cast<uint32_t>(data_ << 16)).f -
        static_cast<float>(value)).i >> 16;
  }

  template<typename T>
  void operator*=(T value) {
    data_ = Sphinx(Sphinx(static_cast<uint32_t>(data_ << 16)).f *
        static_cast<float>(value)).i >> 16;
  }

 private:
  uint16_t data_;
};

template<>
inline bool BFloat16::operator==(const BFloat16 &value) const {
  return data_ == value.data_;
}

template<>
inline void BFloat16::operator=(const BFloat16 &value) {
  data_ = value.data_;
}

}  // namespace mace

namespace std {
inline float fabs(const mace::BFloat16 &value) {
  return fabs(static_cast<float>(value));
}

inline float abs(const mace::BFloat16 &value) {
  return abs(static_cast<float>(value));
}

inline float sqrt(const mace::BFloat16 &value) {
  return sqrt(static_cast<float>(value));
}

inline float log(const mace::BFloat16 &value) {
  return log(static_cast<float>(value));
}

inline float tanh(const mace::BFloat16 &value) {
  return tanh(static_cast<float>(value));
}

inline float exp(const mace::BFloat16 &value) {
  return exp(static_cast<float>(value));
}

inline int ceil(const mace::BFloat16 &value) {
  return ceil(static_cast<float>(value));
}

inline int floor(const mace::BFloat16 &value) {
  return floor(static_cast<float>(value));
}

inline float max(const mace::BFloat16 &a, const float &b) {
  return max(static_cast<float>(a), b);
}

inline float max(const float &a, const mace::BFloat16 &b) {
  return max(a, static_cast<float>(b));
}

inline float min(const mace::BFloat16 &a, const float &b) {
  return min(static_cast<float>(a), b);
}

inline float min(const float &a, const mace::BFloat16 &b) {
  return min(a, static_cast<float>(b));
}

inline float pow(const mace::BFloat16 &a, const mace::BFloat16 &b) {
  return pow(static_cast<float>(a), static_cast<float>(b));
}

inline float pow(const mace::BFloat16 &a, const float &b) {
  return pow(static_cast<float>(a), b);
}

inline float pow(const float &a, const mace::BFloat16 &b) {
  return pow(a, static_cast<float>(b));
}

inline ostream &operator<<(ostream &ss,  // NOLINT
                           const mace::BFloat16 &value) {
  return ss << static_cast<float>(value);
}

}  // namespace std


inline float operator+(const float &a, const mace::BFloat16 &value) {
  return a + static_cast<float>(value);
}

inline float operator-(const float &a, const mace::BFloat16 &value) {
  return a - static_cast<float>(value);
}

inline float operator*(const float &a, const mace::BFloat16 &value) {
  return a * static_cast<float>(value);
}

inline float operator/(const float &a, const mace::BFloat16 &value) {
  return a / static_cast<float>(value);
}

inline void operator+=(float &a, const mace::BFloat16 &value) {  // NOLINT
  a += static_cast<float>(value);
}

inline void operator-=(float &a, const mace::BFloat16 &value) {  // NOLINT
  a -= static_cast<float>(value);
}

inline void operator*=(float &a, const mace::BFloat16 &value) {  // NOLINT
  a *= static_cast<float>(value);
}

inline void operator/=(float &a, const mace::BFloat16 &value) {  // NOLINT
  a /= static_cast<float>(value);
}

#endif  // MACE_ENABLE_BFLOAT16

#endif  // MACE_CORE_BFLOAT16_H_
