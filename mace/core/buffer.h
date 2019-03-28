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

#ifndef MACE_CORE_BUFFER_H_
#define MACE_CORE_BUFFER_H_

#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>

#include "mace/core/allocator.h"
#include "mace/core/types.h"
#include "mace/utils/logging.h"
#include "mace/utils/macros.h"

namespace mace {
namespace core {
enum BufferType {
  BT_BUFFER,
  BT_IMAGE,
};
}  // namespace core

class BufferBase {
 public:
  BufferBase() : size_(0) {}
  explicit BufferBase(index_t size) : size_(size) {}
  virtual ~BufferBase() {}

  virtual core::BufferType buffer_type() const = 0;

  virtual void *buffer() = 0;

  virtual const void *raw_data() const = 0;

  virtual void *raw_mutable_data() = 0;

  virtual MaceStatus Allocate(index_t nbytes) = 0;

  virtual MaceStatus Allocate(const std::vector<size_t> &shape,
                              DataType data_type) = 0;

  virtual void *Map(index_t offset,
                    index_t length,
                    std::vector<size_t> *pitch) const = 0;

  virtual void UnMap(void *mapped_ptr) const = 0;

  virtual void Map(std::vector<size_t> *pitch) = 0;

  virtual void UnMap() = 0;

  virtual MaceStatus Resize(index_t nbytes) = 0;

  virtual void Copy(void *src, index_t offset, index_t length) = 0;

  virtual bool OnHost() const = 0;

  virtual void Clear() = 0;

  virtual void Clear(index_t size) = 0;

  virtual const std::vector<size_t> shape() const = 0;

  virtual index_t offset() const { return 0; }

  template <typename T>
  const T *data() const {
    return reinterpret_cast<const T *>(raw_data());
  }

  template <typename T>
  T *mutable_data() {
    return reinterpret_cast<T *>(raw_mutable_data());
  }

  index_t size() const { return size_; }

 protected:
  index_t size_;
};

class Buffer : public BufferBase {
 public:
  explicit Buffer(Allocator *allocator)
      : BufferBase(0),
        allocator_(allocator),
        buf_(nullptr),
        mapped_buf_(nullptr),
        is_data_owner_(true) {}

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

  core::BufferType buffer_type() const {
    return core::BufferType::BT_BUFFER;
  }

  void *buffer() {
    MACE_CHECK_NOTNULL(buf_);
    return buf_;
  }

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

  MaceStatus Allocate(index_t nbytes) {
    if (nbytes <= 0) {
      return MaceStatus::MACE_SUCCESS;
    }
    MACE_CHECK(is_data_owner_,
               "data is not owned by this buffer, cannot reallocate");
    if (mapped_buf_ != nullptr) {
      UnMap();
    }
    if (buf_ != nullptr) {
      allocator_->Delete(buf_);
    }
    size_ = nbytes;
    return allocator_->New(nbytes, &buf_);
  }

  MaceStatus Allocate(const std::vector<size_t> &shape,
                      DataType data_type) {
    if (shape.empty()) return MaceStatus::MACE_SUCCESS;
    index_t nbytes = std::accumulate(shape.begin(), shape.end(),
                                     1, std::multiplies<size_t>())
        * GetEnumTypeSize(data_type);
    return this->Allocate(nbytes);
  }

  void *Map(index_t offset, index_t length, std::vector<size_t> *pitch) const {
    MACE_CHECK_NOTNULL(buf_);
    MACE_UNUSED(pitch);
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
  }

  void UnMap() {
    UnMap(mapped_buf_);
    mapped_buf_ = nullptr;
  }

  MaceStatus Resize(index_t nbytes) {
    MACE_CHECK(is_data_owner_,
               "data is not owned by this buffer, cannot resize");
    if (nbytes != size_) {
      if (buf_ != nullptr) {
        allocator_->Delete(buf_);
      }
      size_ = nbytes;
      return allocator_->New(nbytes, &buf_);
    }
    return MaceStatus::MACE_SUCCESS;
  }

  void Copy(void *src, index_t offset, index_t length) {
    MACE_CHECK_NOTNULL(mapped_buf_);
    MACE_CHECK(length <= size_, "out of buffer");
    memcpy(mapped_buf_, reinterpret_cast<char*>(src) + offset, length);
  }

  bool OnHost() const { return allocator_->OnHost(); }

  void Clear() {
    Clear(size_);
  }

  void Clear(index_t size) {
    memset(reinterpret_cast<char*>(raw_mutable_data()), 0, size);
  }

  const std::vector<size_t> shape() const {
    MACE_NOT_IMPLEMENTED;
    return {};
  }

 protected:
  Allocator *allocator_;
  void *buf_;
  void *mapped_buf_;
  bool is_data_owner_;

  MACE_DISABLE_COPY_AND_ASSIGN(Buffer);
};

class Image : public BufferBase {
 public:
  explicit Image(Allocator *allocator)
      : BufferBase(0),
        allocator_(allocator),
        buf_(nullptr),
        mapped_buf_(nullptr) {}

  virtual ~Image() {
    if (mapped_buf_ != nullptr) {
      UnMap();
    }
    if (buf_ != nullptr) {
      allocator_->DeleteImage(buf_);
    }
  }

  inline DataType dtype() const {
    MACE_CHECK_NOTNULL(buf_);
    return data_type_;
  }

  core::BufferType buffer_type() const {
    return core::BufferType::BT_IMAGE;
  }

  void *buffer() {
    MACE_CHECK_NOTNULL(buf_);
    return buf_;
  }

  const void *raw_data() const {
    MACE_CHECK_NOTNULL(mapped_buf_);
    return mapped_buf_;
  }

  void *raw_mutable_data() {
    MACE_CHECK_NOTNULL(mapped_buf_);
    return mapped_buf_;
  }

  MaceStatus Allocate(index_t nbytes) {
    MACE_UNUSED(nbytes);
    LOG(FATAL) << "Image should not call this allocate function";
    return MaceStatus::MACE_SUCCESS;
  }

  MaceStatus Allocate(const std::vector<size_t> &shape,
                      DataType data_type) {
    index_t size = std::accumulate(
        shape.begin(), shape.end(), 1, std::multiplies<index_t>()) *
        GetEnumTypeSize(data_type);
    if (mapped_buf_ != nullptr) {
      UnMap();
    }
    if (buf_ != nullptr) {
      allocator_->DeleteImage(buf_);
    }
    size_ = size;
    shape_ = shape;
    data_type_ = data_type;
    return allocator_->NewImage(shape, data_type, &buf_);
  }

  void *Map(index_t offset, index_t length, std::vector<size_t> *pitch) const {
    MACE_UNUSED(offset);
    MACE_UNUSED(length);
    MACE_UNUSED(pitch);
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
  }

  void UnMap() {
    UnMap(mapped_buf_);
    mapped_buf_ = nullptr;
  }

  MaceStatus Resize(index_t size) {
    MACE_UNUSED(size);
    MACE_NOT_IMPLEMENTED;
    return MaceStatus::MACE_SUCCESS;
  }

  void Copy(void *src, index_t offset, index_t length) {
    MACE_UNUSED(src);
    MACE_UNUSED(offset);
    MACE_UNUSED(length);
    MACE_NOT_IMPLEMENTED;
  }

  bool OnHost() const { return allocator_->OnHost(); }

  void Clear() {
    MACE_NOT_IMPLEMENTED;
  }

  void Clear(index_t size) {
    MACE_UNUSED(size);
    MACE_NOT_IMPLEMENTED;
  }

  const std::vector<size_t> shape() const {
    return shape_;
  }

 private:
  Allocator *allocator_;
  std::vector<size_t> shape_;
  DataType data_type_;
  void *buf_;
  void *mapped_buf_;

  MACE_DISABLE_COPY_AND_ASSIGN(Image);
};

class BufferSlice : public BufferBase {
 public:
  BufferSlice()
      : BufferBase(0), buffer_(nullptr), mapped_buf_(nullptr), offset_(0) {}
  BufferSlice(BufferBase *buffer, index_t offset, index_t length)
    : BufferBase(length),
      buffer_(buffer),
      mapped_buf_(nullptr),
      offset_(offset) {
    MACE_CHECK(offset >= 0, "buffer slice offset should >= 0");
    MACE_CHECK(offset + length <= buffer->size(),
               "buffer slice offset + length (",
               offset,
               " + ",
               length,
               ") should <= ",
               buffer->size());
  }
  BufferSlice(const BufferSlice &other)
      : BufferSlice(other.buffer_, other.offset_, other.size_) {}

  virtual ~BufferSlice() {
    if (buffer_ != nullptr && mapped_buf_ != nullptr) {
      UnMap();
    }
  }

  core::BufferType buffer_type() const {
    return core::BufferType::BT_BUFFER;
  }

  void *buffer() {
    MACE_CHECK_NOTNULL(buffer_);
    return buffer_->buffer();
  }

  const void *raw_data() const {
    if (OnHost()) {
      MACE_CHECK_NOTNULL(buffer_);
      return reinterpret_cast<const char*>(buffer_->raw_data()) + offset_;
    } else {
      MACE_CHECK_NOTNULL(mapped_buf_);
      return mapped_buf_;
    }
  }

  void *raw_mutable_data() {
    if (OnHost()) {
      MACE_CHECK_NOTNULL(buffer_);
      return reinterpret_cast<char*>(buffer_->raw_mutable_data()) + offset_;
    } else {
      MACE_CHECK_NOTNULL(mapped_buf_);
      return mapped_buf_;
    }
  }

  MaceStatus Allocate(index_t size) {
    MACE_UNUSED(size);
    LOG(FATAL) << "BufferSlice should not call allocate function";
    return MaceStatus::MACE_SUCCESS;
  }

  MaceStatus Allocate(const std::vector<size_t> &shape,
                      DataType data_type) {
    MACE_UNUSED(shape);
    MACE_UNUSED(data_type);
    LOG(FATAL) << "BufferSlice should not call allocate function";
    return MaceStatus::MACE_SUCCESS;
  }

  void *Map(index_t offset, index_t length, std::vector<size_t> *pitch) const {
    return buffer_->Map(offset_ + offset, length, pitch);
  }

  void UnMap(void *mapped_ptr) const {
    buffer_->UnMap(mapped_ptr);
  }

  void Map(std::vector<size_t> *pitch) {
    MACE_CHECK_NOTNULL(buffer_);
    MACE_CHECK(mapped_buf_ == nullptr, "mapped buf is not null");
    mapped_buf_ = buffer_->Map(offset_, size_, pitch);
  }

  void UnMap() {
    MACE_CHECK_NOTNULL(mapped_buf_);
    buffer_->UnMap(mapped_buf_);
    mapped_buf_ = nullptr;
  }

  MaceStatus Resize(index_t size) {
    MACE_CHECK(size == size_, "resize buffer slice from ", size_,
      " to ", size, " is illegal");
    return MaceStatus::MACE_SUCCESS;
  }

  void Copy(void *src, index_t offset, index_t length) {
    MACE_UNUSED(src);
    MACE_UNUSED(offset);
    MACE_UNUSED(length);
    MACE_NOT_IMPLEMENTED;
  }

  index_t offset() const { return offset_; }

  bool OnHost() const { return buffer_->OnHost(); }

  void Clear() {
    Clear(size_);
  }

  void Clear(index_t size) {
    memset(raw_mutable_data(), 0, size);
  }

  const std::vector<size_t> shape() const {
    MACE_NOT_IMPLEMENTED;
    return {};
  }

 private:
  BufferBase *buffer_;
  void *mapped_buf_;
  index_t offset_;
};

class ScratchBuffer: public Buffer {
 public:
  explicit ScratchBuffer(Allocator *allocator)
    : Buffer(allocator),
      offset_(0) {}

  ScratchBuffer(Allocator *allocator, void *data, index_t size)
    : Buffer(allocator, data, size),
      offset_(0) {}

  virtual ~ScratchBuffer() {}

  MaceStatus GrowSize(const index_t size) {
    if (offset_ + size > size_) {
      VLOG(1) << "Grow scratch size to: " << size;
      MACE_CHECK(offset_ == 0, "scratch is being used, cannot grow size");
      return Resize(size);
    }
    return MaceStatus::MACE_SUCCESS;
  }

  BufferSlice Scratch(index_t size) {
    MACE_CHECK(offset_ + size <= size_,
               "scratch size not enough: ",
               offset_,
               " + ",
               size,
               " > ",
               size_);
    BufferSlice slice(this, offset_, size);
    offset_ += size;
    return slice;
  }

  void Rewind(index_t offset = 0) {
    offset_ = offset;
  }

  index_t offset() const {
    return offset_;
  }

 private:
  index_t offset_;
};

}  // namespace mace

#endif  // MACE_CORE_BUFFER_H_
