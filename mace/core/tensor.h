//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_TENSOR_H_
#define MACE_CORE_TENSOR_H_

#include "mace/core/allocator.h"
#include "mace/core/common.h"
#include "mace/utils/logging.h"
#include "mace/core/types.h"
#include "mace/core/public/mace.h"

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
    CASE(string, SINGLE_ARG(STMTS))                            \
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
  Tensor()
      : alloc_(GetDeviceAllocator(DeviceType::CPU)),
        size_(0),
        dtype_(DT_FLOAT),
        buffer_(nullptr),
        data_(nullptr),
        unused_(false),
        is_image_(false){};

  Tensor(Allocator *alloc, DataType type)
      : alloc_(alloc),
        size_(0),
        dtype_(type),
        buffer_(nullptr),
        data_(nullptr),
        unused_(false),
        is_image_(false){};

  ~Tensor() {
    MACE_CHECK(data_ == nullptr, "Buffer must be unmapped before destroy");
    if (buffer_ != nullptr) {
      MACE_CHECK_NOTNULL(alloc_);
      if (is_image_) {
        alloc_->DeleteImage(buffer_);
      } else {
        alloc_->Delete(buffer_);
      }
    }
  }

  inline DataType dtype() const { return dtype_; }

  inline void SetDtype(DataType dtype) { dtype_ = dtype; }

  inline const vector<index_t> &shape() const { return shape_; }

  inline const vector<size_t> &image_shape() const { return image_shape_; }

  inline const bool is_image() const { return is_image_; }

  inline index_t dim_size() const { return shape_.size(); }

  inline index_t dim(unsigned int index) const {
    MACE_CHECK(index < shape_.size(), "Dim out of range: ",
                                      index, " >= ", shape_.size());
    return shape_[index];
  }

  inline index_t size() const { return size_; }

  inline index_t raw_size() const { return size_ * SizeOfType(); }

  inline const bool unused() const { return unused_; }

  inline int64_t NumElements() const {
    return std::accumulate(shape_.begin(), shape_.end(), 1,
                           std::multiplies<int64_t>());
  }

  inline const bool OnHost() const { return alloc_->OnHost(); }

  /*
   * Map the device buffer as CPU buffer to access the data, unmap must be
   * called later
   */
  inline void Map() const {
    if (!OnHost()) {
      MACE_CHECK(buffer_ != nullptr && data_ == nullptr);
      data_ = alloc_->Map(buffer_, size_ * SizeOfType());
    }
  }

  inline void MapImage(std::vector<size_t> &mapped_image_pitch) const {
    MACE_CHECK(!OnHost() && buffer_ != nullptr && data_ == nullptr);
    data_ = alloc_->MapImage(buffer_, image_shape_, mapped_image_pitch);
  }

  /*
   *  Unmap the device buffer
   */
  inline void Unmap() const {
    if (!OnHost()) {
      MACE_CHECK(buffer_ != nullptr && data_ != nullptr);
      alloc_->Unmap(buffer_, data_);
      data_ = nullptr;
    }
  }

  void *buffer() const { return buffer_; }

  inline const void *raw_data() const {
    void *data = MappedBuffer();
    MACE_CHECK(data != nullptr || size_ == 0,
               "The tensor is of non-zero shape, but its data is not allocated "
               "or mapped yet.");
    return data;
  }

  template <typename T>
  inline const T *data() const {
    return static_cast<const T *>(raw_data());
  }

  inline void *raw_mutable_data() {
    void *data = MappedBuffer();
    MACE_CHECK(data != nullptr || size_ == 0,
               "The tensor is of non-zero shape, but its data is not allocated "
               "or mapped yet.");
    return data;
  }

  template <typename T>
  inline T *mutable_data() {
    return static_cast<T *>(raw_mutable_data());
  }

  inline void Resize(const vector<index_t> &shape) {
    shape_ = shape;
    index_t size = NumElements();
    if (size_ != size || is_image_) {
      size_ = size;
      MACE_CHECK(data_ == nullptr, "Buffer must be unmapped before resize");
      if (is_image_) {
        alloc_->DeleteImage(buffer_);
      } else {
        alloc_->Delete(buffer_);
      }
      is_image_ = false;
      CASES(dtype_, buffer_ = alloc_->New(size_ * sizeof(T)));
    }
  }

  inline void ResizeImage(const vector<index_t> &shape,
                          const std::vector<size_t> &image_shape) {
    shape_ = shape;
    index_t size = NumElements();
    if (size_ != size || !is_image_) {
      size_ = size;
      MACE_CHECK(data_ == nullptr, "Buffer must be unmapped before resize");

      if (is_image_ && !image_shape_.empty()) {
        MACE_ASSERT(image_shape_.size() == 2
                        && image_shape_[0] >= image_shape[0]
                        || image_shape_[1] >= image_shape[1],
                    "image shape not large enough");
      }
      if (!is_image_ && buffer_ != nullptr) {
        alloc_->Delete(buffer_);
      }
      is_image_ = true;
      if (image_shape_.empty()) {
        image_shape_ = image_shape;
        buffer_ = alloc_->NewImage(image_shape, dtype_);
      }
    }
  }

  inline void ResizeLike(const Tensor &other) {
    if (other.is_image()) {
      ResizeImage(other.shape(), other.image_shape());
    } else {
      Resize(other.shape());
    }
  }

  inline void ResizeLike(const Tensor *other) {
    if (other->is_image()) {
      ResizeImage(other->shape(), other->image_shape());
    } else {
      Resize(other->shape());
    }
  }

  inline void AllocateImageMemory(const std::vector<size_t> &image_shape) {
    is_image_ = true;
    if (image_shape_ != image_shape) {
      if (buffer_ != nullptr) {
        alloc_->DeleteImage(buffer_);
      }
      image_shape_ = image_shape;
      buffer_ = alloc_->NewImage(image_shape, dtype_);
    }
  }

  template <typename T>
  inline void Copy(const T *src, index_t size) {
    MACE_CHECK(size == size_, "copy src and dst with different size.");
    CopyBytes(static_cast<const void *>(src), sizeof(T) * size);
  }

  template <typename SrcType, typename DstType>
  inline void CopyWithCast(const SrcType *src, size_t size) {
    MACE_CHECK(static_cast<index_t>(size) == size_,
               "copy src and dst with different size.");
    unique_ptr<DstType[]> buffer(new DstType[size]);
    for (size_t i = 0; i < size; ++i) {
      buffer[i] = static_cast<DstType>(src[i]);
    }
    CopyBytes(static_cast<const void *>(buffer.get()), sizeof(DstType) * size);
  }

  inline void CopyBytes(const void *src, size_t size) {
    MappingGuard map_this(this);
    memcpy(raw_mutable_data(), src, size);
  }

  inline void DebugPrint() const {
    using namespace numerical_chars;
    std::stringstream os;
    for (int i : shape_) {
      os << i << ", ";
    }

    os.str("");
    os.clear();
    MappingGuard guard(this);
    for (int i = 0; i < size_; ++i) {
      if ( i != 0 && i % shape_[3] == 0) {
        os << "\n";
      }
      CASES(dtype_, (os << (this->data<T>()[i]) << ", "));
    }
    LOG(INFO) << os.str();
  }

  inline size_t SizeOfType() const {
    size_t type_size = 0;
    CASES(dtype_, type_size = sizeof(T));
    return type_size;
  }

  inline void Copy(Tensor &other) {
    alloc_ = other.alloc_;
    dtype_ = other.dtype_;
    ResizeLike(other);
    MappingGuard map_other(&other);
    if (is_image_) {
      LOG(FATAL) << "Not support copy image tensor, please use Copy API.";
    } else {
      CopyBytes(other.raw_data(), size_ * SizeOfType());
    }
  }

  inline void MarkUnused() {
    this->unused_ = true;
  }

  class MappingGuard {
   public:
    MappingGuard(const Tensor *tensor) : tensor_(tensor) {
      if (tensor_ != nullptr) {
        if (tensor_->is_image()) {
          tensor_->MapImage(mapped_image_pitch_);
        } else {
          tensor_->Map();
        }
      }
    }
    ~MappingGuard() {
      if (tensor_ != nullptr) tensor_->Unmap();
    }

    inline const vector<size_t> &mapped_image_pitch() const { return mapped_image_pitch_; }

   private:
    const Tensor *tensor_;
    std::vector<size_t> mapped_image_pitch_;
  };

 private:
  inline void *MappedBuffer() const {
    if (OnHost()) {
      return buffer_;
    }
    return data_;
  }

  Allocator *alloc_;
  index_t size_;
  DataType dtype_;
  // Raw buffer, must be mapped as host accessable data before
  // read or write
  void *buffer_;
  // Mapped buffer
  mutable void *data_;
  vector<index_t> shape_;
  // Image for opencl
  bool unused_;
  bool is_image_;
  std::vector<size_t> image_shape_;

  DISABLE_COPY_AND_ASSIGN(Tensor);
};

}  // namespace tensor

#endif  // MACE_CORE_TENSOR_H_
