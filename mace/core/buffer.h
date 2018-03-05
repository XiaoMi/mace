//
// Copyright (c) 2018 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_BUFFER_H_
#define MACE_CORE_BUFFER_H_

#include "mace/core/types.h"
#include "mace/core/allocator.h"
#include <vector>

namespace mace {

class BufferBase {
 public:
  BufferBase() : size_(0) {}
  BufferBase(index_t size) : size_(size) {}
  virtual ~BufferBase() {}

  virtual void *buffer() = 0;

  virtual const void *raw_data() const = 0;

  virtual void *raw_mutable_data() = 0;

  virtual void *Map(index_t offset,
                    index_t length,
                    std::vector<size_t> *pitch) const = 0;

  virtual void UnMap(void *mapped_ptr) const = 0;

  virtual void Map(std::vector<size_t> *pitch) = 0;

  virtual void UnMap() = 0;

  virtual void Resize(index_t size) = 0;

  virtual void Copy(void *src, index_t offset, index_t length) = 0;

  virtual bool OnHost() const = 0;

  virtual index_t offset() const {
    return 0;
  };

  template<typename T>
  const T *data() const {
    return reinterpret_cast<const T *>(raw_data());
  }

  template<typename T>
  T *mutable_data() {
    return reinterpret_cast<T *>(raw_mutable_data());
  }

  index_t size() const {
    return size_;
  }

 protected:
  index_t size_;
};

class Buffer : public BufferBase {
 public:
  Buffer(Allocator *allocator)
    : BufferBase(0),
      allocator_(allocator),
      buf_(nullptr),
      mapped_buf_(nullptr),
      is_data_owner_(true) {}

  Buffer(Allocator *allocator, index_t size)
    : BufferBase(size),
      allocator_(allocator),
      mapped_buf_(nullptr),
      is_data_owner_(true) {
    buf_ = allocator->New(size);
  }

  Buffer(Allocator *allocator, void *data, index_t size)
    : BufferBase(size),
      allocator_(allocator),
      buf_(data),
      mapped_buf_(nullptr),
      is_data_owner_(false) {}

  virtual ~Buffer() {
    if (mapped_buf_ != nullptr) {
      UnMap();
    }
    if (is_data_owner_ && buf_ != nullptr) {
      allocator_->Delete(buf_);
    }
  }

  void *buffer() {
    MACE_CHECK_NOTNULL(buf_);
    return buf_;
  };

  const void *raw_data() const {
    if (OnHost()) {
      MACE_CHECK_NOTNULL(buf_);
      return buf_;
    } else {
      MACE_CHECK_NOTNULL(mapped_buf_);
      return mapped_buf_;
    }
  }

  void *raw_mutable_data() {
    if (OnHost()) {
      MACE_CHECK_NOTNULL(buf_);
      return buf_;
    } else {
      MACE_CHECK_NOTNULL(mapped_buf_);
      return mapped_buf_;
    }
  }

  void *Map(index_t offset, index_t length, std::vector<size_t> *pitch) const {
    MACE_CHECK_NOTNULL(buf_);
    return allocator_->Map(buf_, offset, length);
  }

  void UnMap(void *mapped_ptr) const {
    MACE_CHECK_NOTNULL(buf_);
    MACE_CHECK_NOTNULL(mapped_ptr);
    allocator_->Unmap(buf_, mapped_ptr);
  }

  void Map(std::vector<size_t> *pitch) {
    MACE_CHECK(mapped_buf_ == nullptr, "buf has been already mapped");
    mapped_buf_ = Map(0, size_, pitch);
  };

  void UnMap() {
    UnMap(mapped_buf_);
    mapped_buf_ = nullptr;
  }

  void Resize(index_t size) {
    MACE_CHECK(is_data_owner_,
               "data is not owned by this buffer, cannot resize");
    if (size != size_) {
      if (buf_ != nullptr) {
        allocator_->Delete(buf_);
      }
      size_ = size;
      buf_ = allocator_->New(size);
    }
  }

  void Copy(void *src, index_t offset, index_t length) {
    MACE_CHECK_NOTNULL(mapped_buf_);
    MACE_CHECK(length <= size_, "out of buffer");
    memcpy(mapped_buf_, (char *) src + offset, length);
  }

  bool OnHost() const {
    return allocator_->OnHost();
  }

 private:
  Allocator *allocator_;
  void *buf_;
  void *mapped_buf_;
  bool is_data_owner_;

 DISABLE_COPY_AND_ASSIGN(Buffer);
};

class Image : public BufferBase {
 public:
  Image()
    : BufferBase(0),
      allocator_(GetDeviceAllocator(OPENCL)),
      buf_(nullptr),
      mapped_buf_(nullptr) {}

  Image(std::vector<size_t> shape, DataType data_type)
    : BufferBase(std::accumulate(shape.begin(), shape.end(),
                                 1, std::multiplies<index_t>())
                   * GetEnumTypeSize(data_type)),
      allocator_(GetDeviceAllocator(OPENCL)),
      mapped_buf_(nullptr) {
    shape_ = shape;
    data_type_ = data_type;
    buf_ = allocator_->NewImage(shape, data_type);
  }

  virtual ~Image() {
    if (mapped_buf_ != nullptr) {
      UnMap();
    }
    if (buf_ != nullptr) {
      allocator_->DeleteImage(buf_);
    }
  }

  void *buffer() {
    MACE_CHECK_NOTNULL(buf_);
    return buf_;
  };

  const void *raw_data() const {
    MACE_CHECK_NOTNULL(mapped_buf_);
    return mapped_buf_;
  }

  void *raw_mutable_data() {
    MACE_CHECK_NOTNULL(mapped_buf_);
    return mapped_buf_;
  }

  std::vector<size_t> image_shape() const {
    return shape_;
  }

  void *Map(index_t offset, index_t length, std::vector<size_t> *pitch) const {
    MACE_NOT_IMPLEMENTED;
    return nullptr;
  }

  void UnMap(void *mapped_ptr) const {
    MACE_CHECK_NOTNULL(buf_);
    MACE_CHECK_NOTNULL(mapped_ptr);
    allocator_->Unmap(buf_, mapped_ptr);
  }

  void Map(std::vector<size_t> *pitch) {
    MACE_CHECK_NOTNULL(buf_);
    MACE_CHECK(mapped_buf_ == nullptr, "buf has been already mapped");
    MACE_CHECK_NOTNULL(pitch);
    mapped_buf_ = allocator_->MapImage(buf_, shape_, pitch);
  };

  void UnMap() {
    UnMap(mapped_buf_);
    mapped_buf_ = nullptr;
  };

  void Resize(index_t size) {}

  void Copy(void *src, index_t offset, index_t length) {
    MACE_NOT_IMPLEMENTED;
  }

  bool OnHost() const {
    return allocator_->OnHost();
  }

 private:
  Allocator *allocator_;
  std::vector<size_t> shape_;
  DataType data_type_;
  void *buf_;
  void *mapped_buf_;

 DISABLE_COPY_AND_ASSIGN(Image);
};

class BufferSlice : public BufferBase {
 public:
  BufferSlice() {}
  BufferSlice(BufferBase *buffer, index_t offset, index_t length)
    : BufferBase(buffer->size()),
      buffer_(buffer),
      mapped_buf_(nullptr),
      offset_(offset),
      length_(length) {
    MACE_CHECK(offset >= 0, "buffer slice offset should >= 0");
    MACE_CHECK(offset + length <= size_,
               "buffer slice offset + length (",
               offset,
               " + ",
               length,
               ") should <= ",
               size_);
  }
  BufferSlice(const BufferSlice &other) : BufferSlice(other.buffer_,
                                                      other.offset_,
                                                      other.length_) {}

  ~BufferSlice() {
    if (mapped_buf_ != nullptr) {
      UnMap();
    }
  }

  void *buffer() {
    return buffer_->buffer();
  };

  const void *raw_data() const {
    if (OnHost()) {
      MACE_CHECK_NOTNULL(buffer_);
      return (char *) buffer_->raw_data() + offset_;
    } else {
      MACE_CHECK_NOTNULL(mapped_buf_);
      return mapped_buf_;
    }
  }

  void *raw_mutable_data() {
    MACE_NOT_IMPLEMENTED;
    return nullptr;
  }

  void *Map(index_t offset, index_t length, std::vector<size_t> *pitch) const {
    MACE_NOT_IMPLEMENTED;
    return nullptr;
  }

  void UnMap(void *mapped_ptr) const {
    MACE_NOT_IMPLEMENTED;
  }

  void Map(std::vector<size_t> *pitch) {
    MACE_CHECK_NOTNULL(buffer_);
    MACE_CHECK(mapped_buf_ == nullptr, "mapped buf is not null");
    mapped_buf_ = buffer_->Map(offset_, length_, pitch);
  };

  void UnMap() {
    MACE_CHECK_NOTNULL(mapped_buf_);
    buffer_->UnMap(mapped_buf_);
    mapped_buf_ = nullptr;
  };

  void Resize(index_t size) {
  }

  void Copy(void *src, index_t offset, index_t length) {
    MACE_NOT_IMPLEMENTED;
  }

  index_t offset() const {
    return offset_;
  }

  bool OnHost() const {
    return buffer_->OnHost();
  }

 private:
  BufferBase *buffer_;
  void *mapped_buf_;
  index_t offset_;
  index_t length_;
};

}

#endif // MACE_CORE_BUFFER_H_
