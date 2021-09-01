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

#include "mace/rpcmems/mtk/mtk_rpcmem.h"

#include <dlfcn.h>
#include <sys/mman.h>

#include "mace/utils/logging.h"
#include "third_party/mtk_rpcmem/includes/libion/ion.h"
#include "third_party/mtk_rpcmem/includes/libion_mtk/ion.h"
#include "third_party/mtk_rpcmem/includes/linux/ion_drv.h"

namespace mace {

namespace {
constexpr unsigned int kIonFlagCached = 1;
constexpr unsigned int kIonFlagCachedNeedSync = 2;
}

MtkRpcmem::MtkRpcmem() : ion_handle_(-1),
                         mtk_ion_wrapper_(new MtkIonWrapper()) {
  auto supported = mtk_ion_wrapper_->IsRpcmemSupported();
  if (supported) {
    ion_handle_ = mtk_ion_wrapper_->mt_ion_open(__FILE__);
    if (ion_handle_ >= 0) {
      valid_ = true;
    } else {
      LOG(WARNING) << "MtkRpcmem, Fail to get ion driver handle.";
    }
  }
  valid_detected_ = true;
}

MtkRpcmem::~MtkRpcmem() {
  if (ion_handle_ >= 0) {
    auto err = mtk_ion_wrapper_->ion_close(ion_handle_);
    if (err != 0) {
      LOG(WARNING) << "~MtkRpcmem, Fail to closes ion handle: " << ion_handle_;
    }
    ion_handle_ = -1;
  }
}

void *MtkRpcmem::New(int nbytes) {
  ion_user_handle_t buf_handle;
  if (mtk_ion_wrapper_->ion_alloc_mm(ion_handle_, static_cast<size_t>(nbytes),
                                     0, kIonFlagCached | kIonFlagCachedNeedSync, &buf_handle)) {
    LOG(WARNING) << "Fail to get ion buffer handle, ion_handle_ = "
                 << ion_handle_ << ", nbytes = " << nbytes;
    return nullptr;
  }

  int buf_share_fd = 0;
  auto ret = mtk_ion_wrapper_->ion_share(ion_handle_, buf_handle,
                                         &buf_share_fd);
  if (ret != 0) {
    LOG(WARNING) << "Fail to get ion buffer share handle, ion_handle_ = "
                 << ion_handle_ << ", nbytes = " << nbytes;
    return nullptr;
  }

  void *buf_addr = mtk_ion_wrapper_->ion_mmap(
      ion_handle_, nullptr, static_cast<size_t>(nbytes), PROT_READ | PROT_WRITE,
      MAP_SHARED, buf_share_fd, 0);
  if (buf_addr == nullptr) {
    LOG(WARNING) << "Fail to map fd, ion_handle_ = "
                 << ion_handle_ << ", nbytes = " << nbytes;
    return nullptr;
  }

  handle_addr_map_[buf_share_fd] = buf_addr;
  addr_handle_map_[buf_addr] = buf_share_fd;
  addr_length_map_[buf_addr] = nbytes;
  return buf_addr;
}

void MtkRpcmem::Delete(void *data) {
  MACE_CHECK(addr_handle_map_.count(data) > 0);
  MACE_CHECK(addr_length_map_.count(data) > 0);
  int buf_share_fd = addr_handle_map_.at(data);
  auto size = static_cast<size_t>(addr_length_map_.at(data));

  handle_addr_map_.erase(buf_share_fd);
  addr_handle_map_.erase(data);
  addr_length_map_.erase(data);

  // 1. Get handle of ION_IOC_SHARE from ion_user_handle_t
  ion_user_handle_t buf_handle = 0;
  if (mtk_ion_wrapper_->ion_import(ion_handle_, buf_share_fd, &buf_handle)) {
    LOG(WARNING) << "Fail to get import share buffer fd";
    return;
  }
  // 2. Free for IMPORT ref cnt
  if (mtk_ion_wrapper_->ion_free(ion_handle_, buf_handle)) {
    LOG(WARNING) << "Fail to free ion buffer (free ion_import ref cnt)";
    return;
  }
  // 3. Unmap virtual memory address
  if (mtk_ion_wrapper_->ion_munmap(ion_handle_, data, size)) {
    LOG(WARNING) << "Fail to get unmap virtual memory";
    return;
  }
  // 4. Close share buffer fd
  if (mtk_ion_wrapper_->ion_share_close(ion_handle_, buf_share_fd)) {
    LOG(WARNING) << "Fail to close share buffer fd";
    return;
  }
  // 5. Pair of ion_alloc_mm
  if (mtk_ion_wrapper_->ion_free(ion_handle_, buf_handle)) {
    LOG(WARNING) << "Fail to free ion buffer (free ion_alloc_mm ref cnt)";
  }
}

int MtkRpcmem::ToFd(void *data) {
  MACE_CHECK(addr_handle_map_.count(data) > 0,
             "ToFd failed, data: ", data);
  return addr_handle_map_.at(data);
}

int MtkRpcmem::SyncCacheStart(void *data) {
  // invalid the cache
  MACE_CHECK(addr_handle_map_.count(data) > 0);
  MACE_CHECK(addr_length_map_.count(data) > 0);
  int buf_share_fd = addr_handle_map_.at(data);
  auto size = static_cast<size_t>(addr_length_map_.at(data));

  struct ion_sys_data sys_data;
  ion_user_handle_t ion_user_handle;
  auto import_ret = mtk_ion_wrapper_->ion_import(ion_handle_, buf_share_fd,
                                                 &ion_user_handle);
  MACE_CHECK(0 == import_ret, "Fail to get import share buffer fd");

  sys_data.sys_cmd = ION_SYS_CACHE_SYNC;
  sys_data.cache_sync_param.handle = ion_user_handle;
  sys_data.cache_sync_param.sync_type = ION_CACHE_INVALID_BY_RANGE;
  sys_data.cache_sync_param.va = data;
  sys_data.cache_sync_param.size = size;

  return mtk_ion_wrapper_->ion_custom_ioctl(ion_handle_, ION_CMD_SYSTEM,
                                            &sys_data);
}

int MtkRpcmem::SyncCacheEnd(void *data) {
  // flush the cache
  MACE_CHECK(addr_handle_map_.count(data) > 0);
  MACE_CHECK(addr_length_map_.count(data) > 0);
  int buf_share_fd = addr_handle_map_.at(data);
  auto size = static_cast<size_t>(addr_length_map_.at(data));

  struct ion_sys_data sys_data;
  ion_user_handle_t ion_user_handle;
  auto import_ret = mtk_ion_wrapper_->ion_import(ion_handle_, buf_share_fd,
                                                 &ion_user_handle);
  MACE_CHECK(0 == import_ret, "Fail to get import share buffer fd");

  sys_data.sys_cmd = ION_SYS_CACHE_SYNC;
  sys_data.cache_sync_param.handle = ion_user_handle;
  sys_data.cache_sync_param.sync_type = ION_CACHE_FLUSH_BY_RANGE;
  sys_data.cache_sync_param.va = data;
  sys_data.cache_sync_param.size = size;

  return mtk_ion_wrapper_->ion_custom_ioctl(ion_handle_, ION_CMD_SYSTEM,
                                            &sys_data);
}

RpcmemType MtkRpcmem::GetRpcmemType() {
  return RpcmemType::ION_MTK;
}

}  // namespace mace
