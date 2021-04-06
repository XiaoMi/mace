/*
 *  ion.h
 *
 * Memory Allocator functions for ion
 *
 *   Copyright 2011 Google, Inc
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef __MTK_ION_H
#define __MTK_ION_H


__BEGIN_DECLS

typedef int ion_user_handle_t;

// Allocate buffer from multimedia heap.
int ion_alloc_mm(int fd, size_t len, size_t align,
                 unsigned int flags, ion_user_handle_t *handle);
// Map virtual address of ION buffer
void* ion_mmap(int fd, void *addr, size_t length, int prot, int flags, int share_fd, off_t offset);
// UnMap virtual address of ION buffer
int ion_munmap(int fd, void *addr, size_t length);
// Close share fd of ION buffer.
int ion_share_close(int fd, int share_fd);
// ION custom IOCTL
int ion_custom_ioctl(int fd, unsigned int cmd, void *arg);
int mt_ion_open(const char *name);


__END_DECLS

#endif /* __MTK_ION_H */
