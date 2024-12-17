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

#ifndef MACE_CORE_TENSOR_H_
#define MACE_CORE_TENSOR_H_

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "mace/core/memory/buffer.h"
#include "mace/core/ops/ops_utils.h"
#include "mace/core/runtime/runtime.h"
#include "mace/core/types.h"
#include "mace/utils/logging.h"

#ifdef MACE_ENABLE_NEON
// Avoid over-bound accessing memory
#define MACE_EXTRA_BUFFER_PAD_SIZE 64
#else
#define MACE_EXTRA_BUFFER_PAD_SIZE 0
#endif

namespace mace {

#define MACE_SINGLE_ARG(...) __VA_ARGS__
#define MACE_CASE(TYPE, STATEMENTS)   \
  case DataTypeToEnum<TYPE>::value: { \
    typedef TYPE T;                   \
    STATEMENTS;                       \
    break;                            \
  }

#if defined(MACE_ENABLE_NEON) && defined(__ANDROID__)   \
    || defined(MACE_ENABLE_FP16)
#define MACE_TYPE_ENUM_SWITCH_CASE_FLOAT16(STATEMENTS)     \
  MACE_CASE(float16_t, MACE_SINGLE_ARG(STATEMENTS))
#else
#define MACE_TYPE_ENUM_SWITCH_CASE_FLOAT16(STATEMENTS)
#endif

#ifdef MACE_ENABLE_BFLOAT16
#define MACE_TYPE_ENUM_SWITCH_CASE_BFLOAT16(STATEMENTS) \
  MACE_CASE(BFloat16, MACE_SINGLE_ARG(STATEMENTS))
#else
#define MACE_TYPE_ENUM_SWITCH_CASE_BFLOAT16(STATEMENTS)
#endif  // MACE_ENABLE_BFLOAT16

#ifdef MACE_ENABLE_MTK_APU
#define MACE_TYPE_ENUM_SWITCH_CASE_INT16(STATEMENTS) \
  MACE_CASE(int16_t, MACE_SINGLE_ARG(STATEMENTS))
#else
#define MACE_TYPE_ENUM_SWITCH_CASE_INT16(STATEMENTS)
#endif  // MACE_ENABLE_MTK_APU

#if MACE_ENABLE_OPENCL
#define MACE_TYPE_ENUM_SWITCH_CASE_OPENCL(STATEMENTS)   \
  MACE_CASE(half, MACE_SINGLE_ARG(STATEMENTS))
#else
#define MACE_TYPE_ENUM_SWITCH_CASE_OPENCL(STATEMENTS)
#endif  // MACE_ENABLE_OPENCL

#define MACE_TYPE_ENUM_SWITCH(                                     \
    TYPE_ENUM, STATEMENTS, INVALID_STATEMENTS, DEFAULT_STATEMENTS) \
  switch (TYPE_ENUM) {                                             \
    MACE_CASE(float, MACE_SINGLE_ARG(STATEMENTS))                  \
    MACE_CASE(uint8_t, MACE_SINGLE_ARG(STATEMENTS))                \
    MACE_CASE(uint16_t, MACE_SINGLE_ARG(STATEMENTS))               \
    MACE_CASE(int32_t, MACE_SINGLE_ARG(STATEMENTS))                \
    MACE_CASE(bool, MACE_SINGLE_ARG(STATEMENTS))                   \
    MACE_TYPE_ENUM_SWITCH_CASE_FLOAT16(STATEMENTS)                 \
    MACE_TYPE_ENUM_SWITCH_CASE_BFLOAT16(STATEMENTS)                \
    MACE_TYPE_ENUM_SWITCH_CASE_INT16(STATEMENTS)                   \
    MACE_TYPE_ENUM_SWITCH_CASE_OPENCL(STATEMENTS)                  \
    case DT_INVALID:                                               \
      INVALID_STATEMENTS;                                          \
      break;                                                       \
    default:                                                       \
      DEFAULT_STATEMENTS;                                          \
      break;                                                       \
  }

// `TYPE_ENUM` will be converted to template `T` in `STATEMENTS`
#define MACE_RUN_WITH_TYPE_ENUM(TYPE_ENUM, STATEMENTS)                       \
  MACE_TYPE_ENUM_SWITCH(TYPE_ENUM, STATEMENTS, LOG(FATAL) << "Invalid type"; \
         , LOG(FATAL) << "Unknown type: " << TYPE_ENUM;)

class Tensor {
  friend class Runtime;
 public:
  explicit Tensor(Runtime *runtime, DataType dt, MemoryType mem_type,
                  const std::vector<index_t> &shape = std::vector<index_t>(),
                  const bool is_weight = false, const std::string name = "",
                  const BufferContentType content_type = IN_OUT_CHANNEL)
      : shape_(shape),
        runtime_(runtime),
        buffer_(std::make_shared<Buffer>(mem_type, dt)),
        unused_(false),
        name_(name),
        is_weight_(is_weight),
        scale_(0.f),
        zero_point_(0),
        minval_(0.f),
        maxval_(0.f),
        data_format_(DataFormat::NONE),
        content_type_(content_type) {
    MACE_CHECK(dtype() != DataType::DT_INVALID);
  }

  explicit Tensor(Runtime *runtime, DataType dt,
                  const std::vector<index_t> &shape = std::vector<index_t>(),
                  const bool is_weight = false, const std::string name = "",
                  const BufferContentType content_type = IN_OUT_CHANNEL)
      : shape_(shape),
        runtime_(runtime),
        buffer_(std::make_shared<Buffer>(runtime->GetUsedMemoryType(), dt)),
        unused_(false),
        name_(name),
        is_weight_(is_weight),
        scale_(0.f),
        zero_point_(0),
        minval_(0.f),
        maxval_(0.f),
        data_format_(DataFormat::NONE),
        content_type_(content_type) {
    MACE_CHECK(dtype() != DataType::DT_INVALID);
  }

  ~Tensor() {}

  std::string name() const;
  DataType dtype() const;
  void SetDtype(DataType dtype);
  bool unused() const;
  const std::vector<index_t> &shape() const;
  std::vector<index_t> max_shape() const;
  index_t max_size() const;
  index_t raw_max_size() const;
  void SetShapeConfigured(const std::vector<index_t> &shape_configured);
  void SetContentType(BufferContentType content_type,
                      unsigned int content_param = 0);
  void GetContentType(BufferContentType *content_type,
                      unsigned int *content_param) const;
  const std::vector<index_t> &buffer_shape() const;
  index_t dim_size() const;
  index_t dim(unsigned int index) const;
  index_t size() const;
  index_t raw_size() const;
  MemoryType memory_type() const;
  void set_data_format(DataFormat data_format);
  DataFormat data_format() const;
  index_t buffer_offset() const;
  Runtime *GetCurRuntime() const;

  template<typename T>
  const T *data() const {
    MACE_CHECK_NOTNULL(buffer_);
    return buffer_->data<T>();
  }

  template<typename T>
  T *mutable_data() {
    MACE_CHECK_NOTNULL(buffer_);
    return buffer_->mutable_data<T>();
  }

  const void *raw_data() const;
  void *raw_mutable_data();

  template<typename T>
  const T *memory() const {
    MACE_CHECK_NOTNULL(buffer_);
    return buffer_->memory<T>();
  }

  template<typename T>
  T *mutable_memory() const {
    MACE_CHECK_NOTNULL(buffer_);
    return buffer_->mutable_memory<T>();
  }

  void MarkUnused();
  void Clear();
  void Reshape(const std::vector<index_t> &shape);
  MaceStatus Resize(const std::vector<index_t> &shape);
  void to4dim();
  // Make this tensor reuse other tensor's buffer.
  // This tensor has the same dtype, shape and buffer shape.
  // It could be reshaped later (with buffer shape unchanged).
  void ReuseTensorBuffer(const Tensor &other);
  MaceStatus ResizeLike(const Tensor &other);
  MaceStatus ResizeLike(const Tensor *other);
  void CopyBytes(const void *src, size_t bytes);

  template<typename T>
  void Copy(const T *src, index_t length) {
    MACE_CHECK(length == size(), "copy src and dst with different size.");
    CopyBytes(static_cast<const void *>(src), sizeof(T) * length);
  }

  void Copy(const Tensor &other);
  size_t SizeOfType() const;
  Buffer *UnderlyingBuffer() const;
  void DebugPrint() const;

  void Map(bool wait_for_finish) const;
  void UnMap() const;
  class MappingGuard {
   public:
    explicit MappingGuard(const Tensor *tensor, bool wait_for_finish = true);
    explicit MappingGuard(MappingGuard &&other);
    ~MappingGuard();

   private:
    const Tensor *tensor_;
    MACE_DISABLE_COPY_AND_ASSIGN(MappingGuard);
  };

  bool is_weight() const;
  float scale() const;
  int32_t zero_point() const;

  // hexagon now uses min/max instead of scale and zero
  float minval() const;
  float maxval() const;
  void SetScale(float scale);
  void SetZeroPoint(int32_t zero_point);
  void SetIsWeight(bool is_weight);
  void SetMinVal(float minval);
  void SetMaxVal(float maxval);

 private:
  std::vector<index_t> shape_;
  Runtime *runtime_;
  std::vector<index_t> shape_configured_;
  std::shared_ptr<Buffer> buffer_;
  bool unused_;
  std::string name_;
  bool is_weight_;
  float scale_;
  int32_t zero_point_;
  float minval_;
  float maxval_;
  DataFormat data_format_;  // used for 4D input/output tensor
  BufferContentType content_type_;
  unsigned int content_param_;  // TODO(luxuhui): remove it

  MACE_DISABLE_COPY_AND_ASSIGN(Tensor);
};

}  // namespace mace

#endif  // MACE_CORE_TENSOR_H_
