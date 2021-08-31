// Copyright 2021 The MACE Authors. All Rights Reserved.
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

#include "mace/rpcmems/dmabufheap/dma_buf_heap_rpcmem.h"

#include <dlfcn.h>
#include <sys/mman.h>
#include <unistd.h>

#include "mace/utils/logging.h"

namespace mace {

DmaBufHeapRpcmem::DmaBufHeapRpcmem() : buffer_allocator_(nullptr),
                                       dma_buf_wrapper_(new DmaBufWrapper()) {
  auto supported = dma_buf_wrapper_->IsRpcmemSupported();
  if (supported) {
    buffer_allocator_ = dma_buf_wrapper_->CreateDmabufHeapBufferAllocator();
    if (buffer_allocator_ != nullptr) {
      valid_ = true;
    } else {
      LOG(WARNING) << "DmaBufHeapRpcmem, Fail to get buffer allocator";
    }
  }
  valid_detected_ = true;
}

DmaBufHeapRpcmem::~DmaBufHeapRpcmem() {
  if (buffer_allocator_ != nullptr) {
    dma_buf_wrapper_->FreeDmabufHeapBufferAllocator(buffer_allocator_);
  }
}

void *DmaBufHeapRpcmem::New(int nbytes) {
  int buf_shared_fd = 0;
  buf_shared_fd = dma_buf_wrapper_->DmabufHeapAllocSystem(buffer_allocator_, true, nbytes, 0, 0);
  if (buf_shared_fd == 0) {
    LOG(WARNING) << "Fail to get ion buffer share fd, buf_shared_fd = "
              << buf_shared_fd << ", nbytes = " << nbytes;
    return nullptr;
  }
  void *buf_addr = mmap(nullptr, static_cast<size_t>(nbytes), PROT_READ | PROT_WRITE, MAP_SHARED, buf_shared_fd, 0);
  if (buf_addr == nullptr) {
    LOG(WARNING) << "Fail to map fd, buf_shared_fd = "
                 << buf_shared_fd << ", nbytes = " << nbytes;
    return nullptr;
  }

  handle_addr_map_[buf_shared_fd] = buf_addr;
  addr_handle_map_[buf_addr] = buf_shared_fd;
  addr_length_map_[buf_addr] = nbytes;
  return buf_addr;
}

void DmaBufHeapRpcmem::Delete(void *data) {
  MACE_CHECK(addr_handle_map_.count(data) > 0);
  MACE_CHECK(addr_length_map_.count(data) > 0);
  int buf_share_fd = addr_handle_map_.at(data);
  auto size = static_cast<size_t>(addr_length_map_.at(data));

  handle_addr_map_.erase(buf_share_fd);
  addr_handle_map_.erase(data);
  addr_length_map_.erase(data);

  munmap(data, size);
  close(buf_share_fd);
}

int DmaBufHeapRpcmem::ToFd(void *data) {
  MACE_CHECK(addr_handle_map_.count(data) > 0,
             "ToFd failed, data: ", data);
  return addr_handle_map_.at(data);
}

int DmaBufHeapRpcmem::SyncCacheStart(void *data) {
  // Invalid the cache
  MACE_CHECK(addr_handle_map_.count(data) > 0);
  MACE_CHECK(addr_length_map_.count(data) > 0);
  int buf_share_fd = addr_handle_map_.at(data);

  return dma_buf_wrapper_->DmabufHeapCpuSyncStart(buffer_allocator_, buf_share_fd,
                                                  kSyncReadWrite, nullptr, nullptr);
}

int DmaBufHeapRpcmem::SyncCacheEnd(void *data) {
  // Flush the cache
  MACE_CHECK(addr_handle_map_.count(data) > 0);
  MACE_CHECK(addr_length_map_.count(data) > 0);
  int buf_share_fd = addr_handle_map_.at(data);

  return dma_buf_wrapper_->DmabufHeapCpuSyncEnd(buffer_allocator_, buf_share_fd,
                                                kSyncReadWrite, nullptr, nullptr);
}

RpcmemType DmaBufHeapRpcmem::GetRpcmemType() {
  return RpcmemType::DMA_BUF_HEAP;
}

}  // namespace mace
