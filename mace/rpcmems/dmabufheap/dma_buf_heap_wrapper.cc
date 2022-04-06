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


#include "mace/rpcmems/dmabufheap/dma_buf_heap_wrapper.h"

#include <dlfcn.h>
#include <unistd.h>

#include "mace/utils/logging.h"

namespace mace {

#define MACE_LD_ION_SYMBOL(handle, T, ptr, symbol)     \
  ptr =  reinterpret_cast<T *>(dlsym(handle, symbol)); \
  if (ptr == nullptr) {                                \
    return;                                            \
  }

DmaBufWrapper::DmaBufWrapper() : valid_(false),
                                 dma_buf_(nullptr),
                                 CreateDmabufHeapBufferAllocator_(nullptr),
                                 FreeDmabufHeapBufferAllocator_(nullptr),
                                 DmabufHeapAlloc_(nullptr),
                                 DmabufHeapAllocSystem_(nullptr),
                                 MapDmabufHeapNameToIonHeap_(nullptr),
                                 DmabufHeapCpuSyncStart_(nullptr),
                                 DmabufHeapCpuSyncEnd_(nullptr),
                                 CheckIonSupport_(nullptr) {
  dma_buf_ = dlopen("libdmabufheap.so", RTLD_LAZY | RTLD_LOCAL);
  if (!dma_buf_) {
    LOG(WARNING) << "Load libdmabufheap.so failed";
    return;
  }
  MACE_LD_ION_SYMBOL(dma_buf_, FuncCreateDmabufHeapBufferAllocator,
                     CreateDmabufHeapBufferAllocator_, "CreateDmabufHeapBufferAllocator");
  MACE_LD_ION_SYMBOL(dma_buf_, FuncFreeDmabufHeapBufferAllocator,
                     FreeDmabufHeapBufferAllocator_, "FreeDmabufHeapBufferAllocator");
  MACE_LD_ION_SYMBOL(dma_buf_, FuncDmabufHeapAlloc, DmabufHeapAlloc_, "DmabufHeapAlloc");
  MACE_LD_ION_SYMBOL(dma_buf_, FuncDmabufHeapAllocSystem, DmabufHeapAllocSystem_, "DmabufHeapAllocSystem");
  MACE_LD_ION_SYMBOL(dma_buf_, FuncMapDmabufHeapNameToIonHeap,
                     MapDmabufHeapNameToIonHeap_, "MapDmabufHeapNameToIonHeap");
  MACE_LD_ION_SYMBOL(dma_buf_, FuncDmabufHeapCpuSyncStart, DmabufHeapCpuSyncStart_,
                     "DmabufHeapCpuSyncStart");
  MACE_LD_ION_SYMBOL(dma_buf_, FuncDmabufHeapCpuSyncEnd, DmabufHeapCpuSyncEnd_,
                     "DmabufHeapCpuSyncEnd");
  valid_ = true;
}

DmaBufWrapper::~DmaBufWrapper() {
  if (dma_buf_) {
    dlclose(dma_buf_);
  }
}

bool DmaBufWrapper::IsRpcmemSupported() {
  return valid_;
}

BufferAllocator* DmaBufWrapper::CreateDmabufHeapBufferAllocator() {
  return CreateDmabufHeapBufferAllocator_();
}

void DmaBufWrapper::FreeDmabufHeapBufferAllocator(BufferAllocator* buffer_allocator) {
  return FreeDmabufHeapBufferAllocator_(buffer_allocator);
}

int DmaBufWrapper::DmabufHeapAlloc(BufferAllocator* buffer_allocator, const char* heap_name, size_t len,
                                   unsigned int heap_flags, size_t legacy_align) {
  return DmabufHeapAlloc_(buffer_allocator, heap_name, len, heap_flags, legacy_align);
}

int DmaBufWrapper::DmabufHeapAllocSystem(BufferAllocator* buffer_allocator, bool cpu_access, size_t len,
                                         unsigned int heap_flags, size_t legacy_align) {
  return DmabufHeapAllocSystem_(buffer_allocator, cpu_access, len,
                                heap_flags, legacy_align);
}

int DmaBufWrapper::MapDmabufHeapNameToIonHeap(BufferAllocator* buffer_allocator, const char* heap_name,
                                              const char* ion_heap_name, unsigned int ion_heap_flags,
                                              unsigned int legacy_ion_heap_mask, unsigned legacy_ion_heap_flags) {
return MapDmabufHeapNameToIonHeap_(buffer_allocator, heap_name,
                                   ion_heap_name, ion_heap_flags,
                                   legacy_ion_heap_mask, legacy_ion_heap_flags);
}

int DmaBufWrapper::DmabufHeapCpuSyncStart(BufferAllocator* buffer_allocator, unsigned int dmabuf_fd,
                                          SyncType sync_type, int (*legacy_ion_cpu_sync)(int, int, void *),
                                          void *legacy_ion_custom_data) {
return DmabufHeapCpuSyncStart_(buffer_allocator, dmabuf_fd,
                               sync_type, legacy_ion_cpu_sync,
                               legacy_ion_custom_data);
}

int DmaBufWrapper::DmabufHeapCpuSyncEnd(BufferAllocator* buffer_allocator, unsigned int dmabuf_fd,
                                        SyncType sync_type, int (*legacy_ion_cpu_sync)(int, int, void*),
                                        void* legacy_ion_custom_data) {
  return DmabufHeapCpuSyncEnd_(buffer_allocator, dmabuf_fd,
                               sync_type, legacy_ion_cpu_sync,
                               legacy_ion_custom_data);
}

bool DmaBufWrapper::CheckIonSupport() {
  return CheckIonSupport_();
}

}  // namespace mace
