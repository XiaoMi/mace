//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_TENSOR_H_
#define MACE_CORE_TENSOR_H_

#include "mace/core/buffer.h"
#include "mace/core/preallocated_pooled_allocator.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/types.h"
#include "mace/public/mace.h"
#include "mace/utils/logging.h"

namespace mace {

#define SINGLE_ARG(...) __VA_ARGS__
#define CASE(TYPE, STMTS)             \
  case DataTypeToEnum<TYPE>::value: { \
    typedef TYPE T;                   \
    STMTS;                            \
    break;                            \
  }

#define CASES_WITH_DEFAULT(TYPE_ENUM, STMTS, INVALID, DEFAULT) \
  switch (TYPE_ENUM) {                                         \
    CASE(half, SINGLE_ARG(STMTS))                              \
    CASE(float, SINGLE_ARG(STMTS))                             \
    CASE(double, SINGLE_ARG(STMTS))                            \
    CASE(int32_t, SINGLE_ARG(STMTS))                           \
    CASE(uint8_t, SINGLE_ARG(STMTS))                           \
    CASE(uint16_t, SINGLE_ARG(STMTS))                          \
    CASE(int16_t, SINGLE_ARG(STMTS))                           \
    CASE(int8_t, SINGLE_ARG(STMTS))                            \
    CASE(std::string, SINGLE_ARG(STMTS))                       \
    CASE(int64_t, SINGLE_ARG(STMTS))                           \
    CASE(bool, SINGLE_ARG(STMTS))                              \
    case DT_INVALID:                                           \
      INVALID;                                                 \
      break;                                                   \
    default:                                                   \
      DEFAULT;                                                 \
      break;                                                   \
  }

#define CASES(TYPE_ENUM, STMTS)                                      \
  CASES_WITH_DEFAULT(TYPE_ENUM, STMTS, LOG(FATAL) << "Type not set"; \
                     , LOG(FATAL) << "Unexpected type: " << TYPE_ENUM;)

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
}

class Tensor {
 public:
  Tensor(Allocator *alloc, DataType type)
    : allocator_(alloc),
      dtype_(type),
      buffer_(nullptr),
      is_buffer_owner_(true),
      name_("") {};

  Tensor(BufferBase *buffer, DataType dtype)
    : dtype_(dtype),
      buffer_(buffer),
      is_buffer_owner_(false),
      name_("") {}

  Tensor(const BufferSlice &buffer_slice, DataType dtype)
    : dtype_(dtype),
      buffer_slice_(buffer_slice),
      is_buffer_owner_(false),
      name_("") {
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

  inline const std::vector<index_t> &shape() const { return shape_; }

  inline index_t dim_size() const { return shape_.size(); }

  inline index_t dim(unsigned int index) const {
    MACE_CHECK(index < shape_.size(), "Dim out of range: ",
               index, " >= ", shape_.size());
    return shape_[index];
  }

  inline index_t size() const {
    return std::accumulate(shape_.begin(), shape_.end(), 1,
                           std::multiplies<int64_t>());
  }

  inline index_t raw_size() const {
    return size() * SizeOfType();
  }

  inline bool has_opencl_image() const {
    return buffer_ != nullptr && !buffer_->OnHost()
      && typeid(*buffer_) == typeid(Image);
  }

  inline bool has_opencl_buffer() const {
    return buffer_ != nullptr && !buffer_->OnHost()
      && !has_opencl_image();
  }

  inline cl::Image *opencl_image() const {
    MACE_CHECK(has_opencl_image(), "do not have image");
    return static_cast<cl::Image*>(buffer_->buffer());
  }

  inline cl::Buffer *opencl_buffer() const {
    MACE_CHECK(has_opencl_buffer(), "do not have opencl buffer");
    return static_cast<cl::Buffer*>(buffer_->buffer());
  }

  inline index_t buffer_offset() const {
    return buffer_->offset();
  }

  inline const void *raw_data() const {
    MACE_CHECK(buffer_ != nullptr, "buffer is null");
    return buffer_->raw_data();
  }

  template<typename T>
  inline const T *data() const {
    MACE_CHECK(buffer_ != nullptr, "buffer is null");
    return buffer_->data<T>();
  }

  inline void *raw_mutable_data() {
    MACE_CHECK(buffer_ != nullptr, "buffer is null");
    return buffer_->raw_mutable_data();
  }

  template<typename T>
  inline T *mutable_data() {
    MACE_CHECK(buffer_ != nullptr, "buffer is null");
    return static_cast<T *>(buffer_->raw_mutable_data());
  }

  inline void Reshape(const std::vector<index_t> &shape) {
    shape_ = shape;
    MACE_CHECK(raw_size() <= buffer_->size());
  }

  inline void Resize(const std::vector<index_t> &shape) {
    shape_ = shape;
    if (buffer_ != nullptr) {
      MACE_CHECK(!has_opencl_image(), "Cannot resize image, use ResizeImage.");
      buffer_->Resize(raw_size());
    } else {
      buffer_ = new Buffer(allocator_, raw_size());
      is_buffer_owner_ = true;
    }
  }

  inline void ResizeImage(const std::vector<index_t> &shape,
                          const std::vector<size_t> &image_shape) {
    shape_ = shape;
    if (buffer_ == nullptr) {
      buffer_ = new Image(image_shape, dtype_);
      is_buffer_owner_ = true;
    } else {
      MACE_CHECK(has_opencl_image(), "Cannot ResizeImage buffer, use Resize.");
      Image *image = dynamic_cast<Image*>(buffer_);
      MACE_CHECK(shape[0] <= image->image_shape()[0]
                   && shape[1] <= image->image_shape()[1],
                 "tensor (source op ",
                 name_,
                 "): current image shape: ",
                 image->image_shape()[0],
                 ", ",
                 image->image_shape()[1],
                 " < resize tensor shape: ",
                 shape[0],
                 ", ",
                 shape[1]);
    }
  }

  inline void ResizeLike(const Tensor &other) {
    ResizeLike(&other);
  }

  inline void ResizeLike(const Tensor *other) {
    if (other->has_opencl_image()) {
      if (is_buffer_owner_ && buffer_ != nullptr && !has_opencl_image()) {
        delete buffer_;
        buffer_ = nullptr;
      }
      ResizeImage(other->shape(),
                  dynamic_cast<Image *>(other->UnderlyingBuffer())->image_shape());
    } else {
      if (is_buffer_owner_ && buffer_ != nullptr && has_opencl_image()) {
        delete buffer_;
        buffer_ = nullptr;
      }
      Resize(other->shape());
    }
  }

  inline void CopyBytes(const void *src, size_t size) {
    MappingGuard guard(this);
    memcpy(buffer_->raw_mutable_data(), src, size);
  }

  template<typename T>
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
    CASES(dtype_, type_size = sizeof(T));
    return type_size;
  }

  inline BufferBase *UnderlyingBuffer() const {
    return buffer_;
  }

  inline void SetSourceOpName(const std::string name) {
    name_ = name;
  }

  inline void DebugPrint() const {
    using namespace numerical_chars;
    std::stringstream os;
    for (index_t i : shape_) {
      os << i << ", ";
    }

    os.str("");
    os.clear();
    MappingGuard guard(this);
    for (int i = 0; i < size(); ++i) {
      if (i != 0 && i % shape_[3] == 0) {
        os << "\n";
      }
      CASES(dtype_, (os << (this->data<T>()[i]) << ", "));
    }
    LOG(INFO) << "Tensor size: [" << dim(0) << ", " << dim(1) << ", "
              << dim(2) << ", " << dim(3) << "], content:\n" << os.str();
  }

  class MappingGuard {
   public:
    MappingGuard(const Tensor *tensor) : tensor_(tensor) {
      if (tensor_ != nullptr) {
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

   DISABLE_COPY_AND_ASSIGN(MappingGuard);
  };

 private:
  Allocator *allocator_;
  DataType dtype_;
  std::vector<index_t> shape_;
  BufferBase *buffer_;
  BufferSlice buffer_slice_;
  bool is_buffer_owner_;
  std::string name_;

 DISABLE_COPY_AND_ASSIGN(Tensor);
};

}  // namespace tensor

#endif  // MACE_CORE_TENSOR_H_
