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


#include "mace/rpcmems/mtk/mtk_ion_wrapper.h"

#include <dlfcn.h>

#include "third_party/mtk_rpcmem/includes/libion_mtk/ion.h"

namespace mace {

#define MACE_LD_ION_SYMBOL(handle, T, ptr, symbol)     \
  ptr =  reinterpret_cast<T *>(dlsym(handle, symbol)); \
  if (ptr == nullptr) {                                \
    return;                                            \
  }

MtkIonWrapper::MtkIonWrapper() : valid_(false),
                                 h_ion_(nullptr),
                                 h_ion_mtk_(nullptr),
                                 mt_ion_open_(nullptr),
                                 ion_alloc_mm_(nullptr),
                                 ion_mmap_(nullptr),
                                 ion_import_(nullptr),
                                 ion_munmap_(nullptr),
                                 ion_share_close_(nullptr),
                                 ion_custom_ioctl_(nullptr),
                                 ion_close_(nullptr),
                                 ion_share_(nullptr),
                                 ion_free_(nullptr) {
  // Load symbol from libion_mtk.so
  h_ion_mtk_ = dlopen("libion_mtk.so", RTLD_LAZY | RTLD_LOCAL);
  if (!h_ion_mtk_) {
    return;
  }
  MACE_LD_ION_SYMBOL(h_ion_mtk_, FuncMtIonOpen, mt_ion_open_, "mt_ion_open");
  MACE_LD_ION_SYMBOL(h_ion_mtk_, FuncIonAllocMm, ion_alloc_mm_, "ion_alloc_mm");
  MACE_LD_ION_SYMBOL(h_ion_mtk_, FuncIonMmap, ion_mmap_, "ion_mmap");
  MACE_LD_ION_SYMBOL(h_ion_mtk_, FuncIonImport, ion_import_, "ion_import");
  MACE_LD_ION_SYMBOL(h_ion_mtk_, FuncIonMunmap, ion_munmap_, "ion_munmap");
  MACE_LD_ION_SYMBOL(h_ion_mtk_, FuncIonShareClose, ion_share_close_,
                     "ion_share_close");
  MACE_LD_ION_SYMBOL(h_ion_mtk_, FuncIonCustomIoctl, ion_custom_ioctl_,
                     "ion_custom_ioctl");

  // Load symbol from libion.so
  h_ion_ = dlopen("libion.so", RTLD_LAZY | RTLD_LOCAL);
  if (!h_ion_) {
    return;
  }
  MACE_LD_ION_SYMBOL(h_ion_, FuncIonClose, ion_close_, "ion_close");
  MACE_LD_ION_SYMBOL(h_ion_, FuncIonShare, ion_share_, "ion_share");
  MACE_LD_ION_SYMBOL(h_ion_, FuncIonFree, ion_free_, "ion_free");

  valid_ = true;
}

MtkIonWrapper::~MtkIonWrapper() {
  if (h_ion_mtk_) {
    dlclose(h_ion_mtk_);
  }
  if (h_ion_) {
    dlclose(h_ion_);
  }
}

bool MtkIonWrapper::IsRpcmemSupported() {
  return valid_;
}

int MtkIonWrapper::mt_ion_open(const char *name) {
  return mt_ion_open_(name);
}

int MtkIonWrapper::ion_alloc_mm(int fd, size_t len, size_t align,
                                unsigned int flags, ion_user_handle_t *handle) {
  return ion_alloc_mm_(fd, len, align, flags, handle);
}

void *MtkIonWrapper::ion_mmap(int fd, void *addr, size_t length, int prot,
                              int flags, int share_fd, off_t offset) {
  return ion_mmap_(fd, addr, length, prot, flags, share_fd, offset);
}

int MtkIonWrapper::ion_import(int fd, int share_fd, ion_user_handle_t *handle) {
  return ion_import_(fd, share_fd, handle);
}

int MtkIonWrapper::ion_munmap(int fd, void *addr, size_t length) {
  return ion_munmap_(fd, addr, length);
}

int MtkIonWrapper::ion_share_close(int fd, int share_fd) {
  return ion_share_close_(fd, share_fd);
}

int MtkIonWrapper::ion_custom_ioctl(int fd, unsigned int cmd, void *arg) {
  return ion_custom_ioctl_(fd, cmd, arg);
}

int MtkIonWrapper::ion_close(int fd) {
  return ion_close_(fd);
}

int MtkIonWrapper::ion_share(int fd, ion_user_handle_t handle, int *share_fd) {
  return ion_share_(fd, handle, share_fd);
}

int MtkIonWrapper::ion_free(int fd, ion_user_handle_t handle) {
  return ion_free_(fd, handle);
}

}  // namespace mace

