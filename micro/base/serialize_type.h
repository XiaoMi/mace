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

#ifndef MICRO_BASE_SERIALIZE_TYPE_H_
#define MICRO_BASE_SERIALIZE_TYPE_H_

#include <stdint.h>

#include "micro/include/public/micro.h"

namespace micro {

#ifdef MACE_OFFSET_USE_16
typedef uint16_t offset_size_t;
#else
typedef uint32_t offset_size_t;
#endif  // MACE_OFFSET_USE_16

template<typename T>
struct SerialArray {
  offset_size_t size_;
  offset_size_t offset_;
  SerialArray() : size_(0), offset_(0) {}
};

struct SerialString {
  offset_size_t packed_length_;
  offset_size_t offset_;
  SerialString() : packed_length_(0), offset_(0) {}
};

struct SerialBytes {
  offset_size_t packed_length_;
  offset_size_t offset_;
  SerialBytes() : packed_length_(0), offset_(0) {}
};

typedef float SerialFloat;
typedef int32_t SerialInt32;
typedef uint32_t SerialUint32;
typedef uint32_t SerialBool;
typedef int32_t SerialInt16;
typedef uint32_t SerialUint16;
typedef int32_t SerialInt8;
typedef uint32_t SerialUint8;

#ifndef MACE_DECLARE_OBJECT_FUNC
#define MACE_DECLARE_OBJECT_FUNC(T, OBJECT_NAME) \
  T OBJECT_NAME() const;
#endif  // MACE_DECLARE_OBJECT_FUNC

#ifndef MACE_DEFINE_OBJECT_FUNC
#define MACE_DEFINE_OBJECT_FUNC(CLASS_NAME, T, OBJECT_NAME) \
  T CLASS_NAME::OBJECT_NAME() const {                       \
    return OBJECT_NAME##_;                                  \
  }
#endif  // MACE_DEFINE_OBJECT_FUNC

#ifndef MACE_MACE_DECLARE_PTR_FUNC
#define MACE_DECLARE_PTR_FUNC(T, OBJECT_NAME) \
  const T *OBJECT_NAME() const;
#endif  // MACE_DECLARE_PTR_FUNC

#ifndef MACE_DEFINE_PTR_FUNC
#define MACE_DEFINE_PTR_FUNC(CLASS_NAME, T, OBJECT_NAME) \
  const T *CLASS_NAME::OBJECT_NAME() const {             \
    return &OBJECT_NAME##_;                              \
  }
#endif  // MACE_DEFINE_PTR_FUNC

#ifndef MACE_DECLARE_ARRAY_FUNC
#define MACE_DECLARE_ARRAY_FUNC(T, OBJECT_NAME) \
  T OBJECT_NAME(uint32_t index) const;          \
  uint32_t OBJECT_NAME##_size() const
#endif  // MACE_DECLARE_ARRAY_FUNC

#ifndef MACE_DECLARE_ARRAY_BASE_PTR_FUNC
#define MACE_DECLARE_ARRAY_BASE_PTR_FUNC(T, OBJECT_NAME) \
  const T * OBJECT_NAME() const
#endif  // MACE_DECLARE_ARRAY_BASE_PTR_FUNC

#ifndef MACE_DEFINE_ARRAY_BASE_PTR_FUNC
#define MACE_DEFINE_ARRAY_BASE_PTR_FUNC(                               \
          CLASS_NAME, T, OBJECT_NAME, ARRAY_NAME)                      \
  const T *CLASS_NAME::OBJECT_NAME() const {                           \
    const T *array = reinterpret_cast<const T *>(                      \
        reinterpret_cast<const uint8_t *>(this) + ARRAY_NAME.offset_); \
    return array;                                                      \
  }
#endif  // MACE_DEFINE_ARRAY_BASE_PTR_FUNC

#ifndef MACE_DEFINE_ARRAY_FUNC
#define MACE_DEFINE_ARRAY_FUNC(CLASS_NAME, T, OBJECT_NAME, ARRAY_NAME) \
  T CLASS_NAME::OBJECT_NAME(uint32_t index) const {                    \
    const T *array = reinterpret_cast<const T *>(                      \
        reinterpret_cast<const uint8_t *>(this) + ARRAY_NAME.offset_); \
    return *(array + index);                                           \
  }                                                                    \
  uint32_t CLASS_NAME::OBJECT_NAME##_size() const {                    \
    return ARRAY_NAME.size_;                                           \
  }
#endif  // MACE_DEFINE_ARRAY_FUNC

#ifndef MACE_DECLARE_PTR_ARRAY_FUNC
#define MACE_DECLARE_PTR_ARRAY_FUNC(T, OBJECT_NAME) \
  const T *OBJECT_NAME(uint32_t index) const;       \
  uint32_t OBJECT_NAME##_size() const
#endif  // MACE_DECLARE_PTR_ARRAY_FUNC

#ifndef MACE_DEFINE_PTR_ARRAY_FUNC
#define MACE_DEFINE_PTR_ARRAY_FUNC(CLASS_NAME, T, OBJECT_NAME, ARRAY_NAME) \
  const T *CLASS_NAME::OBJECT_NAME(uint32_t index) const {                 \
    const T *array = reinterpret_cast<const T *>(                          \
        reinterpret_cast<const uint8_t *>(this) + ARRAY_NAME.offset_);     \
    return (array + index);                                                \
  }                                                                        \
                                                                           \
  uint32_t CLASS_NAME::OBJECT_NAME##_size() const {                        \
    return ARRAY_NAME.size_;                                               \
  }
#endif  // MACE_DEFINE_PTR_ARRAY_FUNC

#ifndef MACE_DECLARE_STRING_FUNC
#define MACE_DECLARE_STRING_FUNC(OBJECT_NAME) \
  const char *OBJECT_NAME() const;
#endif  // MACE_DECLARE_STRING_FUNC

#ifndef MACE_DEFINE_STRING_FUNC
#define MACE_DEFINE_STRING_FUNC(CLASS_NAME, OBJECT_NAME, STRING_NAME)    \
  const char *CLASS_NAME::OBJECT_NAME() const {                          \
    if (STRING_NAME.packed_length_ == 0) {                               \
      return NULL;                                                       \
    } else {                                                             \
      return reinterpret_cast<const char *>(this) + STRING_NAME.offset_; \
    }                                                                    \
  }
#endif  // MACE_DEFINE_STRING_FUNC

#ifndef MACE_DECLARE_BYTES_FUNC
#define MACE_DECLARE_BYTES_FUNC(OBJECT_NAME) \
  const uint8_t *OBJECT_NAME() const;        \
  uint32_t OBJECT_NAME##_size() const
#endif  // MACE_DECLARE_BYTES_FUNC

#ifndef MACE_DEFINE_BYTES_FUNC
#define MACE_DEFINE_BYTES_FUNC(CLASS_NAME, OBJECT_NAME, BYTES_NAME)        \
  const uint8_t *CLASS_NAME::OBJECT_NAME() const {                         \
    if (BYTES_NAME.packed_length_ == 0) {                                  \
        return NULL;                                                       \
    } else {                                                               \
      return reinterpret_cast<const uint8_t *>(this) + BYTES_NAME.offset_; \
    }                                                                      \
  }                                                                        \
                                                                           \
  uint32_t CLASS_NAME::OBJECT_NAME##_size() const {                        \
    return BYTES_NAME.packed_length_;                                      \
  }
#endif  // MACE_DEFINE_BYTES_FUNC

#ifndef MACE_DECLARE_STRING_ARRAY_FUNC
#define MACE_DECLARE_STRING_ARRAY_FUNC(OBJECT_NAME)   \
  const char *OBJECT_NAME(uint32_t index) const; \
  uint32_t OBJECT_NAME##_size() const
#endif

#ifndef MACE_DEFINE_STRING_ARRAY_FUNC
#define MACE_DEFINE_STRING_ARRAY_FUNC(CLASS_NAME, OBJECT_NAME, ARRAY_NAME) \
  const char *CLASS_NAME::OBJECT_NAME(uint32_t index) const {              \
    const SerialString *array = reinterpret_cast<const SerialString *>(    \
        reinterpret_cast<const char *>(this) + ARRAY_NAME.offset_);        \
    const SerialString *serial_str = array + index;                        \
    const char *str = reinterpret_cast<const char *>(serial_str) +         \
        serial_str->offset_;                                               \
    return str;                                                            \
  }                                                                        \
                                                                           \
  uint32_t CLASS_NAME::OBJECT_NAME##_size() const {                        \
    return ARRAY_NAME.size_;                                               \
  }
#endif  // MACE_DEFINE_STRING_ARRAY_FUNC

}  // namespace micro

#endif  // MICRO_BASE_SERIALIZE_TYPE_H_
