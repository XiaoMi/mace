/*
 * Copyright (C) 2020 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef BUFFER_ALLOCATOR_H_
#define BUFFER_ALLOCATOR_H_

#include "dmabufheap-defs.h"
#include <stdbool.h>
#include <sys/types.h>

class BufferAllocator;

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

#endif
