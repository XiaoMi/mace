//
// Copyright (c) 2018 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_ARM_CONV_WINOGRAD_H_
#define MACE_KERNELS_ARM_CONV_WINOGRAD_H_

#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "mace/core/types.h"

namespace mace {
namespace kernels {

void WinoGradConv3x3s1(const float *input,
                       const float *filter,
                       const index_t batch,
                       const index_t in_height,
                       const index_t in_width,
                       const index_t in_channels,
                       const index_t out_channels,
                       float *output);

void WinoGradConv3x3s1(const float *input,
                       const float *filter,
                       const index_t batch,
                       const index_t in_height,
                       const index_t in_width,
                       const index_t in_channels,
                       const index_t out_channels,
                       float *transformed_input,
                       float *transformed_filter,
                       float *transformed_output,
                       bool is_filter_transformed,
                       float *output);

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_ARM_CONV_WINOGRAD_H_
