/*==============================================================================
  Copyright (c) 2012-2013 Qualcomm Technologies, Inc.
  All rights reserved. Qualcomm Proprietary and Confidential.
==============================================================================*/

#ifndef RPCMEM_H
#define RPCMEM_H

#include <stdint.h>

/**
 * RPCMEM_DEFAULT_HEAP
 * Dynamicaly select the heap to use.  This should be ok for most usecases.
 */
#define RPCMEM_DEFAULT_HEAP -1

/**
 * RPCMEM HEAP IDs
 * SYSTEM HEAP:
 *   - non-contiguous physical memory
 *   - for sub-systems with SMMU
 *   - recommended for HVX/CDSPs
 * CONTIG HEAP:
 *   - Contiguous physical memory
 *   - limited memory
 *   - for sub-systems without SMMU (ex. sDSP and mDSP)
 */
#define RPCMEM_HEAP_ID_SYSTEM         (25)
#define RPCMEM_HEAP_ID_CONTIG         (22)
#define RPCMEM_HEAP_ID_SECURE         (9)
#define RPCMEM_HEAP_ID_SYSTEM_CONTIG  (21)

/**
 * RPCMEM_DEFAULT_FLAGS should allocate memory with the same properties
 * as the ION_FLAG_CACHED flag
 */
#ifdef ION_FLAG_CACHED
#define RPCMEM_DEFAULT_FLAGS ION_FLAG_CACHED
#else
#define RPCMEM_DEFAULT_FLAGS 1
#endif

/**
 * RPCMEM_FLAG_UNCACHED
 * ION_FLAG_CACHED should be defined as 1
 */
#define RPCMEM_FLAG_UNCACHED 0
#define RPCMEM_FLAG_CACHED RPCMEM_DEFAULT_FLAGS

/**
 * examples:
 *
 * heap 22, uncached, 1kb
 *    rpcmem_alloc(22, 0, 1024);
 *    rpcmem_alloc(22, RPCMEM_FLAG_UNCACHED, 1024);
 *
 * heap 21, cached, 2kb
 *    rpcmem_alloc(21, RPCMEM_FLAG_CACHED, 2048);
 *    #include <ion.h>
 *    rpcmem_alloc(21, ION_FLAG_CACHED, 2048);
 *
 * just give me the defaults, 2kb
 *    rpcmem_alloc(RPCMEM_DEFAULT_HEAP, RPCMEM_DEFAULT_FLAGS, 2048);
 *    rpcmem_alloc_def(2048);
 *
 * give me the default flags, but from heap 18, 4kb
 *    rpcmem_alloc(18, RPCMEM_DEFAULT_FLAGS, 4096);
 *
 */
#define ION_SECURE_FLAGS    ((1 << 31) | (1 << 19))

/**
 * To flag start/end for rpcmem_sync_cache
 */
#define RPCMEM_SYNC_START 0
#define RPCMEM_SYNC_END   1

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rpcmem rpcmem;
struct rpcmem {
  void *lst;  // QList*
  void *mt;   // pthread_mutex_t*
  int ionfd;
  int flag;
  int ionversion;
};

/**
 * call once to initialize the library
 * NOTE: rpcmem_init is now thread safe
 */
void rpcmem_init(rpcmem *rm);
/**
 * call once for cleanup
 * NOTE: rpcmem_deinit is now thread safe
 */
void rpcmem_deinit(rpcmem *rm);

/**
 * Allocate via ION a buffer of size
 * @heapid, the heap id to use
 * @flags, ion flags to use to when allocating
 * @size, the buffer size to allocate
 * @retval, 0 on failure, pointer to buffer on success
 *
 * For example:
 *    buf = rpcmem_alloc(rm, RPCMEM_DEFAULT_HEAP, RPCMEM_DEFAULT_FLAGS, size);
 */

void* rpcmem_alloc(rpcmem *rm, int heapid, uint32_t flags, int size);

/**
 * allocate with default settings
 */
 #if !defined(WINNT) && !defined (_WIN32_WINNT)
__attribute__((unused))
#endif
static __inline void* rpcmem_alloc_def(rpcmem *rm, int size) {
  return rpcmem_alloc(rm, RPCMEM_DEFAULT_HEAP, RPCMEM_DEFAULT_FLAGS, size);
}

/**
 * free buffer, ignores invalid buffers
 */
void rpcmem_free(rpcmem *rm, void* po);

/**
 * returns associated fd
 */
int rpcmem_to_fd(rpcmem *rm, void* po);

/**
 * cache coherency management
 */
int rpcmem_sync_cache(rpcmem *rm, void* po, uint32_t flags);

#ifdef __cplusplus
}
#endif

/** these are deprecated
 */
#define RPCMEM_HEAP_DEFAULT     0x80000000
#define RPCMEM_HEAP_NOREG       0x40000000
#define RPCMEM_HEAP_UNCACHED    0x20000000
#define RPCMEM_HEAP_NOVA        0x10000000
#define RPCMEM_HEAP_NONCOHERENT 0x08000000

#endif //RPCMEM_H
