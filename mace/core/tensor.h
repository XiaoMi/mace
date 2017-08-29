//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_TENSOR_H_
#define MACE_CORE_TENSOR_H_

#include "mace/core/common.h"
#include "mace/proto/mace.pb.h"
#include "mace/core/allocator.h"
#include "mace/core/types.h"

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
    CASE(float, SINGLE_ARG(STMTS))                             \
    CASE(double, SINGLE_ARG(STMTS))                            \
    CASE(int32, SINGLE_ARG(STMTS))                             \
    CASE(uint8, SINGLE_ARG(STMTS))                             \
    CASE(uint16, SINGLE_ARG(STMTS))                            \
    CASE(int16, SINGLE_ARG(STMTS))                             \
    CASE(int8, SINGLE_ARG(STMTS))                              \
    CASE(string, SINGLE_ARG(STMTS))                            \
    CASE(int64, SINGLE_ARG(STMTS))                             \
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


class Tensor {
 public:
  Tensor()
      : alloc_(cpu_allocator()),
        size_(0), dtype_(DT_FLOAT), data_(nullptr) {};

  Tensor(Allocator* a, DataType type)
      : alloc_(a), size_(0), dtype_(type), data_(nullptr) {};

  ~Tensor() {
    if (alloc_ && data_.get()) {
      data_.reset();
    }
  }

  inline DataType dtype() const { return dtype_; }

  inline const vector<TIndex>& shape() const { return shape_; }

  inline TIndex dim_size() { return shape_.size(); }

  inline TIndex size() const { return size_; }

  inline const void* raw_data() const {
    CHECK(data_.get() || size_ == 0);
    return data_.get();
  }

  template <typename T>
  inline const T* data() const {
    REQUIRE(
        data_.get() || size_ == 0,
        "The tensor is of non-zero shape, but its data is not allocated yet. ");
    return static_cast<T*>(data_.get());
  }

  inline void* raw_mutable_data() {
    if (data_.get() || size_ == 0) {
      return data_.get();
    } else {
      CASES(dtype_, data_.reset(alloc_->New(size_ * sizeof(T)), [this](void* ptr) {
        alloc_->Delete(ptr);
      }));
      return data_.get();
    }
  }

  template <typename T>
  inline T* mutable_data() {
    if (size_ == 0 || data_.get()) {
      return static_cast<T*>(data_.get());
    }
    return static_cast<T*>(raw_mutable_data());
  }

  inline void Resize(const vector<TIndex>& shape) {
    shape_ = shape;
    TIndex size = NumElements();
    if (size_ != size) {
      size_ = size;
      data_.reset();
    }
  }

  inline void ResizeLike(const Tensor& other) {
    Resize(other.shape());
  }

  inline void ResizeLike(const Tensor* other) {
    Resize(other->shape());
  }

 private:
  inline int64 NumElements() const {
    return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int64>());
  }

  Allocator* alloc_;
  TIndex size_;
  DataType dtype_;
  std::shared_ptr<void> data_;
  vector<TIndex> shape_;
};

} // namespace tensor

#endif //MACE_CORE_TENSOR_H_
