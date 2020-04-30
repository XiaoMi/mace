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

#ifndef MACE_CORE_MEMORY_BUFFER_H_
#define MACE_CORE_MEMORY_BUFFER_H_

#include <vector>

#include "mace/core/memory/allocator.h"
#include "mace/core/types.h"
#include "mace/proto/mace.pb.h"

namespace mace {

class Buffer : public MemInfo {
 public:
  explicit Buffer(
      const MemoryType buffer_mt, DataType dt,
      const std::vector<index_t> buffer_dims = std::vector<index_t>(),
      void *buffer_ptr = nullptr)
      : MemInfo(buffer_mt, dt, buffer_dims), buf_(buffer_ptr), host_(nullptr) {}

  explicit Buffer(const MemInfo &info, void *ptr)
      : MemInfo(info), buf_(ptr), host_(nullptr) {}

  explicit Buffer(const Buffer &buf)
      : MemInfo(buf.mem_type, buf.data_type, buf.dims),
        buf_(buf.buf_), host_(buf.host_) {}

  template<typename T>
  const T *data() const {
    return reinterpret_cast<const T *>(host_);
  }

  template<typename T>
  T *mutable_data() {
    return reinterpret_cast<T *>(host_);
  }

  template<typename T>
  const T *memory() const {
    return reinterpret_cast<const T *>(buf_);
  }

  template<typename T>
  T *mutable_memory() {
    return reinterpret_cast<T *>(buf_);
  }

  MaceStatus Resize(const std::vector<index_t> &buffer_dims) {
    dims = buffer_dims;
    return MaceStatus::MACE_SUCCESS;
  }

  void SetBuf(void *buf) {
    buf_ = buf;
  }

  void SetHost(void *host) {
    host_ = host;
  }

  virtual index_t offset() {
    return 0;
  }

 private:
  void *buf_;
  void *host_;
};

}  // namespace mace

#endif  // MACE_CORE_MEMORY_BUFFER_H_
