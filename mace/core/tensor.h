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

#ifndef MACE_CORE_TENSOR_H_
#define MACE_CORE_TENSOR_H_

#include <string>
#include <vector>
#include <functional>

#include "mace/core/buffer.h"
#include "mace/core/preallocated_pooled_allocator.h"
#include "mace/core/types.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/cl2_header.h"
#endif
#include "mace/public/mace.h"
#include "mace/utils/logging.h"

#ifdef MACE_ENABLE_NEON
// Avoid over-bound accessing memory
#define MACE_EXTRA_BUFFER_PAD_SIZE 64
#else
#define MACE_EXTRA_BUFFER_PAD_SIZE 0
#endif

namespace mace {

#define MACE_SINGLE_ARG(...) __VA_ARGS__
#define MACE_CASE(TYPE, STATEMENTS)             \
  case DataTypeToEnum<TYPE>::value: { \
    typedef TYPE T;                   \
    STATEMENTS;                            \
    break;                            \
  }

#ifdef MACE_ENABLE_OPENCL
#define MACE_TYPE_ENUM_SWITCH(                                     \
    TYPE_ENUM, STATEMENTS, INVALID_STATEMENTS, DEFAULT_STATEMENTS) \
  switch (TYPE_ENUM) {                                             \
    MACE_CASE(half, MACE_SINGLE_ARG(STATEMENTS))                   \
    MACE_CASE(float, MACE_SINGLE_ARG(STATEMENTS))                  \
    MACE_CASE(uint8_t, MACE_SINGLE_ARG(STATEMENTS))                \
    MACE_CASE(int32_t, MACE_SINGLE_ARG(STATEMENTS))                \
    case DT_INVALID:                                               \
      INVALID_STATEMENTS;                                          \
      break;                                                       \
    default:                                                       \
      DEFAULT_STATEMENTS;                                          \
      break;                                                       \
  }
#else
#define MACE_TYPE_ENUM_SWITCH(                                     \
    TYPE_ENUM, STATEMENTS, INVALID_STATEMENTS, DEFAULT_STATEMENTS) \
  switch (TYPE_ENUM) {                                             \
    MACE_CASE(float, MACE_SINGLE_ARG(STATEMENTS))                  \
    MACE_CASE(uint8_t, MACE_SINGLE_ARG(STATEMENTS))                \
    MACE_CASE(int32_t, MACE_SINGLE_ARG(STATEMENTS))                \
    case DT_INVALID:                                               \
      INVALID_STATEMENTS;                                          \
      break;                                                       \
    default:                                                       \
      DEFAULT_STATEMENTS;                                          \
      break;                                                       \
  }
#endif

// `TYPE_ENUM` will be converted to template `T` in `STATEMENTS`
#define MACE_RUN_WITH_TYPE_ENUM(TYPE_ENUM, STATEMENTS)                       \
  MACE_TYPE_ENUM_SWITCH(TYPE_ENUM, STATEMENTS, LOG(FATAL) << "Invalid type"; \
         , LOG(FATAL) << "Unknown type: " << TYPE_ENUM;)

namespace numerical_chars {
inline std::ostream &operator<<(std::ostream &os, char c) {
  return std::is_signed<char>::value ? os << static_cast<int>(c)
                                     : os << static_cast<unsigned int>(c);
}

inline std::ostream &operator<<(std::ostream &os, signed char c) {
  return os << static_cast<int>(c);
}

inline std::ostream &operator<<(std::ostream &os, unsigned char c) {
  return os << static_cast<unsigned int>(c);
}
}  // namespace numerical_chars

enum DataFormat { NHWC = 0, NCHW = 1, HWOI = 2, OIHW = 3, HWIO = 4, OHWI = 5 };

class Tensor {
 public:
  Tensor(Allocator *alloc, DataType type)
      : allocator_(alloc),
        dtype_(type),
        buffer_(nullptr),
        is_buffer_owner_(true),
        unused_(false),
        name_(""),
        scale_(0.f),
        zero_point_(0) {}

  Tensor(BufferBase *buffer, DataType dtype)
    : dtype_(dtype),
      buffer_(buffer),
      is_buffer_owner_(false),
      unused_(false),
      name_(""),
      scale_(0.f),
      zero_point_(0) {}

  Tensor(const BufferSlice &buffer_slice, DataType dtype)
      : dtype_(dtype),
        buffer_slice_(buffer_slice),
        is_buffer_owner_(false),
        unused_(false),
        name_(""),
        scale_(0.f),
        zero_point_(0) {
    buffer_ = &buffer_slice_;
  }

  Tensor() : Tensor(GetDeviceAllocator(CPU), DT_FLOAT) {}

  ~Tensor() {
    if (is_buffer_owner_ && buffer_ != nullptr) {
      delete buffer_;
    }
  }

  inline DataType dtype() const { return dtype_; }

  inline void SetDtype(DataType dtype) { dtype_ = dtype; }

  inline bool unused() const { return unused_; }

  inline const std::vector<index_t> &shape() const { return shape_; }

  inline index_t dim_size() const { return shape_.size(); }

  inline index_t dim(unsigned int index) const {
    MACE_CHECK(index < shape_.size(), "Dim out of range: ", index, " >= ",
               shape_.size());
    return shape_[index];
  }

  inline index_t size() const {
    return std::accumulate(shape_.begin(), shape_.end(), 1,
                           std::multiplies<int64_t>());
  }

  inline index_t raw_size() const { return size() * SizeOfType(); }

  inline bool has_opencl_image() const {
    return buffer_ != nullptr && !buffer_->OnHost() &&
           typeid(*buffer_) == typeid(Image);
  }

  inline bool has_opencl_buffer() const {
    return buffer_ != nullptr && !buffer_->OnHost() && !has_opencl_image();
  }

#ifdef MACE_ENABLE_OPENCL
  inline cl::Image *opencl_image() const {
    MACE_CHECK(has_opencl_image(), "do not have image");
    return static_cast<cl::Image *>(buffer_->buffer());
  }

  inline cl::Buffer *opencl_buffer() const {
    MACE_CHECK(has_opencl_buffer(), "do not have opencl buffer");
    return static_cast<cl::Buffer *>(buffer_->buffer());
  }
#endif

  inline index_t buffer_offset() const { return buffer_->offset(); }

  inline const void *raw_data() const {
    MACE_CHECK(buffer_ != nullptr, "buffer is null");
    return buffer_->raw_data();
  }

  template <typename T>
  inline const T *data() const {
    MACE_CHECK_NOTNULL(buffer_);
    return buffer_->data<T>();
  }

  inline void *raw_mutable_data() {
    MACE_CHECK_NOTNULL(buffer_);
    return buffer_->raw_mutable_data();
  }

  template <typename T>
  inline T *mutable_data() {
    MACE_CHECK_NOTNULL(buffer_);
    return static_cast<T *>(buffer_->raw_mutable_data());
  }

  inline void MarkUnused() {
    unused_ = true;
  }

  inline void Clear() {
    MACE_CHECK_NOTNULL(buffer_);
    buffer_->Clear(raw_size());
  }

  inline void Reshape(const std::vector<index_t> &shape) {
    shape_ = shape;
    if (has_opencl_image()) {
      MACE_CHECK(raw_size() <= 4 * buffer_->size());
    } else {
      MACE_CHECK(raw_size() <= buffer_->size());
    }
  }

  inline MaceStatus Resize(const std::vector<index_t> &shape) {
    shape_ = shape;
    image_shape_.clear();
    if (buffer_ != nullptr) {
      MACE_CHECK(!has_opencl_image(), "Cannot resize image, use ResizeImage.");
      if (raw_size() + MACE_EXTRA_BUFFER_PAD_SIZE > buffer_->size()) {
        LOG(WARNING) << "Resize buffer from size " << buffer_->size() << " to "
                     << raw_size() + MACE_EXTRA_BUFFER_PAD_SIZE;
        return buffer_->Resize(raw_size() + MACE_EXTRA_BUFFER_PAD_SIZE);
      }
      return MaceStatus::MACE_SUCCESS;
    } else {
      MACE_CHECK(is_buffer_owner_);
      buffer_ = new Buffer(allocator_);
      return buffer_->Allocate(raw_size() + MACE_EXTRA_BUFFER_PAD_SIZE);
    }
  }

  // Make this tensor reuse other tensor's buffer.
  // This tensor has the same dtype, shape and image_shape.
  // It could be reshaped later (with image shape unchanged).
  inline void ReuseTensorBuffer(const Tensor &other) {
    if (is_buffer_owner_ && buffer_ != nullptr) {
      delete buffer_;
    }
    is_buffer_owner_ = false;
    buffer_ = other.buffer_;
    allocator_ = other.allocator_;
    dtype_ = other.dtype_;
    shape_ = other.shape_;
    image_shape_ = other.image_shape_;
  }

  inline MaceStatus ResizeImage(const std::vector<index_t> &shape,
                                const std::vector<size_t> &image_shape) {
    shape_ = shape;
    image_shape_ = image_shape;
    if (buffer_ == nullptr) {
      MACE_CHECK(is_buffer_owner_);
      buffer_ = new Image();
      return buffer_->Allocate(image_shape, dtype_);
    } else {
      MACE_CHECK(has_opencl_image(), "Cannot ResizeImage buffer, use Resize.");
      Image *image = dynamic_cast<Image *>(buffer_);
      MACE_CHECK(image_shape[0] <= image->image_shape()[0] &&
                     image_shape[1] <= image->image_shape()[1],
                 "tensor (source op ", name_,
                 "): current physical image shape: ", image->image_shape()[0],
                 ", ", image->image_shape()[1], " < logical image shape: ",
                 image_shape[0], ", ", image_shape[1]);
      return MaceStatus::MACE_SUCCESS;
    }
  }

  inline MaceStatus ResizeLike(const Tensor &other) {
    return ResizeLike(&other);
  }

  inline MaceStatus ResizeLike(const Tensor *other) {
    if (other->has_opencl_image()) {
      if (is_buffer_owner_ && buffer_ != nullptr && !has_opencl_image()) {
        delete buffer_;
        buffer_ = nullptr;
      }
      return ResizeImage(other->shape(), other->image_shape_);
    } else {
      if (is_buffer_owner_ && buffer_ != nullptr && has_opencl_image()) {
        delete buffer_;
        buffer_ = nullptr;
      }
      return Resize(other->shape());
    }
  }

  inline void CopyBytes(const void *src, size_t size) {
    MappingGuard guard(this);
    memcpy(buffer_->raw_mutable_data(), src, size);
  }

  template <typename T>
  inline void Copy(const T *src, index_t length) {
    MACE_CHECK(length == size(), "copy src and dst with different size.");
    CopyBytes(static_cast<const void *>(src), sizeof(T) * length);
  }

  inline void Copy(const Tensor &other) {
    dtype_ = other.dtype_;
    ResizeLike(other);
    MappingGuard map_other(&other);
    CopyBytes(other.raw_data(), other.size() * SizeOfType());
  }

  inline size_t SizeOfType() const {
    size_t type_size = 0;
    MACE_RUN_WITH_TYPE_ENUM(dtype_, type_size = sizeof(T));
    return type_size;
  }

  inline BufferBase *UnderlyingBuffer() const { return buffer_; }

  inline void SetSourceOpName(const std::string name) { name_ = name; }

  inline void DebugPrint() const {
    using namespace numerical_chars;  // NOLINT(build/namespaces)
    std::stringstream os;
    os << "Tensor " << name_ << " size: [";
    for (index_t i : shape_) {
      os << i << ", ";
    }
    os << "], content:\n";

    for (int i = 0; i < size(); ++i) {
      if (i != 0 && i % shape_.back() == 0) {
        os << "\n";
      }
      MACE_RUN_WITH_TYPE_ENUM(dtype_, (os << (this->data<T>()[i]) << ", "));
    }
    LOG(INFO) << os.str();
  }

  class MappingGuard {
   public:
    explicit MappingGuard(const Tensor *tensor) : tensor_(tensor) {
      if (tensor_ != nullptr) {
        MACE_CHECK_NOTNULL(tensor_->buffer_);
        tensor_->buffer_->Map(&mapped_image_pitch_);
      }
    }

    MappingGuard(MappingGuard &&other) {
      tensor_ = other.tensor_;
      other.tensor_ = nullptr;
    }

    ~MappingGuard() {
      if (tensor_ != nullptr) tensor_->buffer_->UnMap();
    }

    inline const std::vector<size_t> &mapped_image_pitch() const {
      return mapped_image_pitch_;
    }

   private:
    const Tensor *tensor_;
    std::vector<size_t> mapped_image_pitch_;

    MACE_DISABLE_COPY_AND_ASSIGN(MappingGuard);
  };

  inline float scale() const {
    return scale_;
  }

  inline int32_t zero_point() const {
    return zero_point_;
  }

  inline void SetScale(float scale) {
    scale_ = scale;
  }

  inline void SetZeroPoint(int32_t zero_point) {
    zero_point_ = zero_point;
  }

 private:
  Allocator *allocator_;
  DataType dtype_;
  std::vector<index_t> shape_;
  std::vector<size_t> image_shape_;
  BufferBase *buffer_;
  BufferSlice buffer_slice_;
  bool is_buffer_owner_;
  bool unused_;
  std::string name_;
  float scale_;
  int32_t zero_point_;

  MACE_DISABLE_COPY_AND_ASSIGN(Tensor);
};

}  // namespace mace

#endif  // MACE_CORE_TENSOR_H_
