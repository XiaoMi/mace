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

#ifndef MACE_RPCMEMS_DMABUFHEAP_DMA_BUF_HEAP_WRAPPER_H_
#define MACE_RPCMEMS_DMABUFHEAP_DMA_BUF_HEAP_WRAPPER_H_

#include "third_party/dmabufheap/include/BufferAllocatorWrapper.h"
#include "third_party/dmabufheap/include/dmabufheap-defs.h"

namespace mace {

class DmaBufWrapper {
 public:
  DmaBufWrapper();
  ~DmaBufWrapper();
  bool IsRpcmemSupported();

  BufferAllocator* CreateDmabufHeapBufferAllocator();

  void FreeDmabufHeapBufferAllocator(BufferAllocator* buffer_allocator);

  int DmabufHeapAlloc(BufferAllocator* buffer_allocator, const char* heap_name, size_t len,
                      unsigned int heap_flags, size_t legacy_align);
  int DmabufHeapAllocSystem(BufferAllocator* buffer_allocator, bool cpu_access, size_t len,
                            unsigned int heap_flags, size_t legacy_align);

  int MapDmabufHeapNameToIonHeap(BufferAllocator* buffer_allocator, const char* heap_name,
                                 const char* ion_heap_name, unsigned int ion_heap_flags,
                                 unsigned int legacy_ion_heap_mask, unsigned legacy_ion_heap_flags);

  int DmabufHeapCpuSyncStart(BufferAllocator* buffer_allocator, unsigned int dmabuf_fd,
                             SyncType sync_type, int (*legacy_ion_cpu_sync)(int, int, void *),
                             void *legacy_ion_custom_data);

  int DmabufHeapCpuSyncEnd(BufferAllocator* buffer_allocator, unsigned int dmabuf_fd,
                           SyncType sync_type, int (*legacy_ion_cpu_sync)(int, int, void*),
                           void* legacy_ion_custom_data);

  bool CheckIonSupport();

 private:
  typedef BufferAllocator* FuncCreateDmabufHeapBufferAllocator();

  typedef void FuncFreeDmabufHeapBufferAllocator(BufferAllocator* buffer_allocator);

  typedef int FuncDmabufHeapAlloc(BufferAllocator* buffer_allocator, const char* heap_name, size_t len,
                                  unsigned int heap_flags, size_t legacy_align);
  typedef int FuncDmabufHeapAllocSystem(BufferAllocator* buffer_allocator, bool cpu_access, size_t len,
                                        unsigned int heap_flags, size_t legacy_align);

  typedef int FuncMapDmabufHeapNameToIonHeap(BufferAllocator* buffer_allocator, const char* heap_name,
                                             const char* ion_heap_name, unsigned int ion_heap_flags,
                                             unsigned int legacy_ion_heap_mask, unsigned legacy_ion_heap_flags);

  typedef int FuncDmabufHeapCpuSyncStart(BufferAllocator* buffer_allocator, unsigned int dmabuf_fd,
                                         SyncType sync_type, int (*legacy_ion_cpu_sync)(int, int, void *),
                                         void *legacy_ion_custom_data);

  typedef int FuncDmabufHeapCpuSyncEnd(BufferAllocator* buffer_allocator, unsigned int dmabuf_fd,
                                       SyncType sync_type, int (*legacy_ion_cpu_sync)(int, int, void*),
                                       void* legacy_ion_custom_data);

  typedef bool FuncCheckIonSupport();

 private:
  bool valid_;
  void *dma_buf_;

  FuncCreateDmabufHeapBufferAllocator *CreateDmabufHeapBufferAllocator_;
  FuncFreeDmabufHeapBufferAllocator *FreeDmabufHeapBufferAllocator_;
  FuncDmabufHeapAlloc *DmabufHeapAlloc_;
  FuncDmabufHeapAllocSystem *DmabufHeapAllocSystem_;
  FuncMapDmabufHeapNameToIonHeap *MapDmabufHeapNameToIonHeap_;
  FuncDmabufHeapCpuSyncStart *DmabufHeapCpuSyncStart_;
  FuncDmabufHeapCpuSyncEnd *DmabufHeapCpuSyncEnd_;
  FuncCheckIonSupport *CheckIonSupport_;
};

}  // namespace mace

#endif  // MACE_RPCMEMS_DMABUFHEAP_DMA_BUF_HEAP_WRAPPER_H_
