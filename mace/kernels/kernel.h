// Copyright 2018 Xiaomi, Inc.  All rights reserved.
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

#ifndef MACE_KERNELS_KERNEL_H_
#define MACE_KERNELS_KERNEL_H_

#include "mace/core/op_kernel_context.h"

namespace mace {
namespace kernels {

struct OpKernel {
  explicit OpKernel(OpKernelContext *context): context_(context) {}

  OpKernelContext *context_;
};

}  // namespace kernels
}  // namespace mace
#endif  //  MACE_KERNELS_KERNEL_H_
