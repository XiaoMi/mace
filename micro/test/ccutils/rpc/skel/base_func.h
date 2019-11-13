// Copyright 2020 The MACE Authors. All Rights Reserved.
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


#ifndef MICRO_TEST_CCUTILS_RPC_SKEL_BASE_FUNC_H_
#define MICRO_TEST_CCUTILS_RPC_SKEL_BASE_FUNC_H_

#include <HAP_perf.h>
#include <stdlib.h>

#include "AEEStdErr.h"  // NOLINT
#include "remote.h"  // NOLINT

#ifndef MACE_DEFINE_RANDOM_INPUT
#define MACE_DEFINE_RANDOM_INPUT(NAME)                   \
static remote_handle64 h##NAME = -1;                     \
int NAME##_open(const char *uri, remote_handle64 *h) {   \
  if (h##NAME == -1) {                                   \
    h##NAME = (remote_handle64)(HAP_perf_get_time_us()); \
  }                                                      \
  if (h##NAME == NULL) {                                 \
    h##NAME = -1;                                        \
    return AEE_ENOMEMORY;                                \
  }                                                      \
  *h = h##NAME;                                          \
  return AEE_SUCCESS;                                    \
}                                                        \
int NAME##_close(remote_handle64 h) {                    \
  if (h != h##NAME) {                                    \
    return AEE_EBADPARM;                                 \
  }                                                      \
  if (h##NAME != -1) {                                   \
  }                                                      \
  h##NAME = -1;                                          \
  return AEE_SUCCESS;                                    \
}
#endif  // MACE_DEFINE_RANDOM_INPUT

#ifdef __cplusplus
namespace rpc {
namespace skel {
#endif  // __cplusplus

void FillRandomValue(void *input, const int32_t shape_size);

#ifdef __cplusplus
}  // namespace skel
}  // namespace rpc
#endif  // __cplusplus

#endif  // MICRO_TEST_CCUTILS_RPC_SKEL_BASE_FUNC_H_
