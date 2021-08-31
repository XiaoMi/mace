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

#ifndef DMABUFHEAP_DEF_H_
#define DMABUFHEAP_DEF_H_

// #include <linux/dma-buf.h>

#define DMA_BUF_SYNC_READ      (1 << 0)
#define DMA_BUF_SYNC_WRITE     (2 << 0)
#define DMA_BUF_SYNC_RW        (DMA_BUF_SYNC_READ | DMA_BUF_SYNC_WRITE)

static const char kDmabufSystemHeapName[] = "system";
static const char kDmabufSystemUncachedHeapName[] = "system-uncached";

typedef enum {
    kSyncRead = DMA_BUF_SYNC_READ,
    kSyncWrite = DMA_BUF_SYNC_WRITE,
    kSyncReadWrite = DMA_BUF_SYNC_RW,
} SyncType;

#endif
