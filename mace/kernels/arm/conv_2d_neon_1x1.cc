//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "mace/core/types.h"
#include "mace/kernels/gemm.h"

namespace mace {
namespace kernels {

void Conv2dNeonK1x1S1(const float *input,
                      const float *filter,
                      const index_t batch,
                      const index_t height,
                      const index_t width,
                      const index_t in_channels,
                      const index_t out_channels,
                      float *output) {
  for (index_t b = 0; b < batch; ++b) {
    Gemm(filter,
         input + b * in_channels * height * width,
         1,
         out_channels,
         in_channels,
         height * width,
         output + b * out_channels * height * width);
  }
}

}  // namespace kernels
}  // namespace mace
