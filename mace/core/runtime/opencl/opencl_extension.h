//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_RUNTIME_OPENCL_OPENCL_EXTENSION_H_
#define MACE_CORE_RUNTIME_OPENCL_OPENCL_EXTENSION_H_

#include "mace/core/runtime/opencl/cl2_header.h"

// Adreno extensions
// Adreno performance hints
typedef cl_uint cl_perf_hint;

#define CL_CONTEXT_PERF_HINT_QCOM 0x40C2
#define CL_PERF_HINT_HIGH_QCOM 0x40C3
#define CL_PERF_HINT_NORMAL_QCOM 0x40C4
#define CL_PERF_HINT_LOW_QCOM 0x40C5

// Adreno priority hints
typedef cl_uint cl_priority_hint;

#define CL_PRIORITY_HINT_NONE_QCOM 0
#define CL_CONTEXT_PRIORITY_HINT_QCOM 0x40C9
#define CL_PRIORITY_HINT_HIGH_QCOM 0x40CA
#define CL_PRIORITY_HINT_NORMAL_QCOM 0x40CB
#define CL_PRIORITY_HINT_LOW_QCOM 0x40CC

/* Accepted by clGetKernelWorkGroupInfo */
#define CL_KERNEL_WAVE_SIZE_QCOM 0xAA02
#endif  // MACE_CORE_RUNTIME_OPENCL_OPENCL_EXTENSION_H_
