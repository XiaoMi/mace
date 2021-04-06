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

#ifndef MACE_RPCMEMS_MTK_MTK_ION_WRAPPER_H_
#define MACE_RPCMEMS_MTK_MTK_ION_WRAPPER_H_

#include "third_party/mtk_rpcmem/includes/libion/ion.h"

namespace mace {

class MtkIonWrapper {
 public:
  MtkIonWrapper();
  ~MtkIonWrapper();
  bool IsRpcmemSupported();

  // Function from libion_mtk.so
  int mt_ion_open(const char *name);
  int ion_alloc_mm(int fd, size_t len, size_t align,
                   unsigned int flags, ion_user_handle_t *handle);
  void *ion_mmap(int fd, void *addr, size_t length, int prot, int flags,
                 int share_fd, off_t offset);
  int ion_import(int fd, int share_fd, ion_user_handle_t *handle);
  int ion_munmap(int fd, void *addr, size_t length);
  int ion_share_close(int fd, int share_fd);
  int ion_custom_ioctl(int fd, unsigned int cmd, void *arg);

  // Function from libion.so
  int ion_close(int fd);
  int ion_share(int fd, ion_user_handle_t handle, int *share_fd);
  int ion_free(int fd, ion_user_handle_t handle);

 private:
  // Function from libion_mtk.so
  typedef int FuncMtIonOpen(const char *name);
  typedef int FuncIonAllocMm(int fd, size_t len, size_t align,
                             unsigned int flags, ion_user_handle_t *handle);
  typedef void *FuncIonMmap(int fd, void *addr, size_t length, int prot,
                            int flags, int share_fd, off_t offset);
  typedef int FuncIonImport(int fd, int share_fd, ion_user_handle_t *handle);
  typedef int FuncIonMunmap(int fd, void *addr, size_t length);
  typedef int FuncIonShareClose(int fd, int share_fd);
  typedef int FuncIonCustomIoctl(int fd, unsigned int cmd, void *arg);

  // Function from libion.so
  typedef int FuncIonClose(int fd);
  typedef int FuncIonShare(int fd, ion_user_handle_t handle, int *share_fd);
  typedef int FuncIonFree(int fd, ion_user_handle_t handle);

 private:
  bool valid_;
  void *h_ion_;
  void *h_ion_mtk_;

  // Function ptr from libion_mtk.so
  FuncMtIonOpen *mt_ion_open_;
  FuncIonAllocMm *ion_alloc_mm_;
  FuncIonMmap *ion_mmap_;
  FuncIonImport *ion_import_;
  FuncIonMunmap *ion_munmap_;
  FuncIonShareClose *ion_share_close_;
  FuncIonCustomIoctl *ion_custom_ioctl_;

  // Function ptr from libion.so
  FuncIonClose *ion_close_;
  FuncIonShare *ion_share_;
  FuncIonFree *ion_free_;
};
}  // namespace mace


#endif  // MACE_RPCMEMS_MTK_MTK_ION_WRAPPER_H_
