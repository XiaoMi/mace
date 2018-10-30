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

#ifndef MACE_KERNELS_OPENCL_BUFFER_UTILS_H_
#define MACE_KERNELS_OPENCL_BUFFER_UTILS_H_

#include "mace/core/future.h"
#include "mace/core/op_context.h"
#include "mace/core/tensor.h"
#include "mace/public/mace.h"

namespace mace {
namespace kernels {
namespace opencl {
namespace buffer {

MaceStatus PadInput(OpContext *context,
                    cl::Kernel *kernel,
                    const Tensor *input,
                    const int pad_top,
                    const int pad_left,
                    const bool input_changed,
                    Tensor *padded_input,
                    StatsFuture *future);

}  // namespace buffer
}  // namespace opencl
}  // namespace kernels
}  // namespace mace
#endif  // MACE_KERNELS_OPENCL_BUFFER_UTILS_H_
