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

#include "mace/kernels/opencl/helper.h"

#include <algorithm>
#include <string>
#include <vector>

#include "mace/utils/tuner.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

namespace {
// [(C + 3) / 4 * W, N * H]
void CalInOutputImageShape(const std::vector<index_t> &shape, /* NHWC */
                           std::vector<size_t> *image_shape) {
  MACE_CHECK(shape.size() == 4);
  image_shape->resize(2);
  (*image_shape)[0] = RoundUpDiv4(shape[3]) * shape[2];
  (*image_shape)[1] = shape[0] * shape[1];
}

// [Ic, H * W * (Oc + 3) / 4]
void CalConv2dFilterImageShape(const std::vector<index_t> &shape, /* OIHW */
                               std::vector<size_t> *image_shape) {
  MACE_CHECK(shape.size() == 4);
  image_shape->resize(2);
  (*image_shape)[0] = shape[1];
  (*image_shape)[1] = shape[2] * shape[3] * RoundUpDiv4(shape[0]);
}

// [H * W * M, (Ic + 3) / 4]
void CalDepthwiseConv2dFilterImageShape(
    const std::vector<index_t> &shape, /* MIHW */
    std::vector<size_t> *image_shape) {
  MACE_CHECK(shape.size() == 4);
  image_shape->resize(2);
  (*image_shape)[0] = shape[0] * shape[2] * shape[3];
  (*image_shape)[1] = RoundUpDiv4(shape[1]);
}

// [(size + 3) / 4, 1]
void CalArgImageShape(const std::vector<index_t> &shape,
                      std::vector<size_t> *image_shape) {
  MACE_CHECK(shape.size() == 1);
  image_shape->resize(2);
  (*image_shape)[0] = RoundUpDiv4(shape[0]);
  (*image_shape)[1] = 1;
}

// Only support 3x3 now
// [ (Ic + 3) / 4, 16 * Oc]
void CalWinogradFilterImageShape(
    const std::vector<index_t> &shape, /* Oc, Ic, H, W*/
    std::vector<size_t> *image_shape,
    const int blk_size) {
  MACE_CHECK(shape.size() == 4);
  image_shape->resize(2);
  (*image_shape)[0] = RoundUpDiv4(shape[1]);
  (*image_shape)[1] = (shape[0] * (blk_size + 2) * (blk_size + 2));
}


// [W * C, N * RoundUp<4>(H)]
void CalInOutHeightImageShape(const std::vector<index_t> &shape, /* NHWC */
                              std::vector<size_t> *image_shape) {
  MACE_CHECK(shape.size() == 4);
  image_shape->resize(2);
  (*image_shape)[0] = shape[2] * shape[3];
  (*image_shape)[1] = shape[0] * RoundUpDiv4(shape[1]);
}

// [RoundUp<4>(W) * C, N * H]
void CalInOutWidthImageShape(const std::vector<index_t> &shape, /* NHWC */
                             std::vector<size_t> *image_shape) {
  MACE_CHECK(shape.size() == 4);
  image_shape->resize(2);
  (*image_shape)[0] = RoundUpDiv4(shape[2]) * shape[3];
  (*image_shape)[1] = shape[0] * shape[1];
}

// [Ic * H * W, (Oc + 3) / 4]
void CalWeightHeightImageShape(const std::vector<index_t> &shape, /* OIHW */
                               std::vector<size_t> *image_shape) {
  MACE_CHECK(shape.size() == 4);
  image_shape->resize(2);
  (*image_shape)[0] = shape[1] * shape[2] * shape[3];
  (*image_shape)[1] = RoundUpDiv4(shape[0]);
}

// [(Ic + 3) / 4 * H * W, Oc]
void CalWeightWidthImageShape(const std::vector<index_t> &shape, /* OIHW */
                              std::vector<size_t> *image_shape) {
  MACE_CHECK(shape.size() == 4);
  image_shape->resize(2);
  (*image_shape)[0] = RoundUpDiv4(shape[1]) * shape[2] * shape[3];
  (*image_shape)[1] = shape[0];
}
}  // namespace

void CalImage2DShape(const std::vector<index_t> &shape, /* NHWC */
                     const BufferType type,
                     std::vector<size_t> *image_shape,
                     const int wino_block_size) {
  MACE_CHECK_NOTNULL(image_shape);
  switch (type) {
    case CONV2D_FILTER:
      CalConv2dFilterImageShape(shape, image_shape);
      break;
    case DW_CONV2D_FILTER:
      CalDepthwiseConv2dFilterImageShape(shape, image_shape);
      break;
    case IN_OUT_CHANNEL:
      CalInOutputImageShape(shape, image_shape);
      break;
    case ARGUMENT:
      CalArgImageShape(shape, image_shape);
      break;
    case IN_OUT_HEIGHT:
      CalInOutHeightImageShape(shape, image_shape);
      break;
    case IN_OUT_WIDTH:
      CalInOutWidthImageShape(shape, image_shape);
      break;
    case WINOGRAD_FILTER:
      CalWinogradFilterImageShape(shape, image_shape, wino_block_size);
      break;
    case WEIGHT_HEIGHT:
      CalWeightHeightImageShape(shape, image_shape);
      break;
    case WEIGHT_WIDTH:
      CalWeightWidthImageShape(shape, image_shape);
      break;
    default:
      LOG(FATAL) << "Mace not supported yet.";
  }
}

std::vector<index_t> FormatBufferShape(
    const std::vector<index_t> &buffer_shape,
    const BufferType type) {

  const size_t buffer_shape_size = buffer_shape.size();
  switch (type) {
    case IN_OUT_CHANNEL:
      if (buffer_shape_size == 4) {  // NHWC
        return buffer_shape;
      } else if (buffer_shape_size == 2) {  // NC
        return {buffer_shape[0], 1, 1, buffer_shape[1]};
      } else {
        LOG(FATAL) << "GPU only support 2D or 4D input and output";
      }
    case IN_OUT_HEIGHT:
    case IN_OUT_WIDTH:
      // only used for matmul test
      if (buffer_shape_size == 3) {
        return {buffer_shape[0], buffer_shape[1], buffer_shape[2], 1};
      } else if (buffer_shape_size == 4) {
        return buffer_shape;
      } else {
        LOG(FATAL) << "GPU only support 3D or 4D for IN_OUT_WIDTH "
            "and IN_OUT_HEIGHT";
      }
    default:
      return buffer_shape;
  }
}

std::string DtToCLDt(const DataType dt) {
  switch (dt) {
    case DT_FLOAT:
      return "float";
    case DT_HALF:
      return "half";
    default:
      LOG(FATAL) << "Unsupported data type";
      return "";
  }
}

std::string DtToCLCMDDt(const DataType dt) {
  switch (dt) {
    case DT_FLOAT:
      return "f";
    case DT_HALF:
      return "h";
    default:
      LOG(FATAL) << "Not supported data type for opencl cmd data type";
      return "";
  }
}

std::string DtToUpCompatibleCLDt(const DataType dt) {
  switch (dt) {
    case DT_FLOAT:
    case DT_HALF:
      return "float";
    default:
      LOG(FATAL) << "Unsupported data type";
      return "";
  }
}

std::string DtToUpCompatibleCLCMDDt(const DataType dt) {
  switch (dt) {
    case DT_FLOAT:
    case DT_HALF:
      return "f";
    default:
      LOG(FATAL) << "Not supported data type for opencl cmd data type";
      return "";
  }
}

std::vector<uint32_t> Default3DLocalWS(const uint32_t *gws,
                                       const uint32_t kwg_size) {
  std::vector<uint32_t> lws(4, 0);
  if (kwg_size == 0) {
    lws[0] = lws[1] = lws[2] = 1;
  } else {
    uint64_t cache_size =
        OpenCLRuntime::Global()->device_global_mem_cache_size();
    uint32_t base = std::max<uint32_t>(cache_size / kBaseGPUMemCacheSize, 1);
    lws[1] = std::min<uint32_t>(gws[1], kwg_size);
    lws[2] =
        std::min<uint32_t>(std::min<uint32_t>(gws[2], base), kwg_size / lws[1]);
    const uint32_t lws_size = lws[1] * lws[2];
    lws[0] = std::max<uint32_t>(std::min<uint32_t>(base, kwg_size / lws_size),
                                1);
  }
  return lws;
}

MaceStatus TuningOrRun3DKernel(const cl::Kernel &kernel,
                               const std::string tuning_key,
                               const uint32_t *gws,
                               const std::vector<uint32_t> &lws,
                               StatsFuture *future) {
  auto runtime = OpenCLRuntime::Global();

  auto params_generator = [&]() -> std::vector<std::vector<uint32_t>> {
    const uint32_t kwg_size =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel));
    std::vector<std::vector<uint32_t>> results;
    std::vector<std::vector<uint32_t>> candidates = {
        // TODO(heliangliang): tuning these magic numbers
        {gws[0], gws[1], gws[2], 0},
        {gws[0], gws[1], gws[2] / 8, 0},
        {gws[0], gws[1], gws[2] / 4, 0},
        {gws[0], gws[1], 8, 0},
        {gws[0], gws[1], 4, 0},
        {gws[0], gws[1], 1, 0},
        {gws[0] / 4, gws[1], gws[2], 0},
        {gws[0] / 4, gws[1], gws[2] / 8, 0},
        {gws[0] / 4, gws[1], gws[2] / 4, 0},
        {gws[0] / 4, gws[1], 8, 0},
        {gws[0] / 4, gws[1], 4, 0},
        {gws[0] / 4, gws[1], 1, 0},
        {gws[0] / 8, gws[1], gws[2], 0},
        {gws[0] / 8, gws[1], gws[2] / 8, 0},
        {gws[0] / 8, gws[1], gws[2] / 4, 0},
        {gws[0] / 8, gws[1], 8, 0},
        {gws[0] / 8, gws[1], 4, 0},
        {gws[0] / 8, gws[1], 1, 0},
        {4, gws[1], gws[2], 0},
        {4, gws[1], gws[2] / 8, 0},
        {4, gws[1], gws[2] / 4, 0},
        {4, gws[1], 8, 0},
        {4, gws[1], 4, 0},
        {4, gws[1], 1, 0},
        {1, gws[1], gws[2], 0},
        {1, gws[1], gws[2] / 8, 0},
        {1, gws[1], gws[2] / 4, 0},
        {1, gws[1], 8, 0},
        {1, gws[1], 4, 0},
        {1, gws[1], 1, 0},
    };
    for (auto &ele : candidates) {
      const uint32_t tmp = ele[0] * ele[1] * ele[2];
      if (0 < tmp && tmp <= kwg_size) {
        results.push_back(ele);
      }
    }
    return results;
  };
  cl::Event event;
  auto func = [&](const std::vector<uint32_t> &params, Timer *timer,
                  std::vector<uint32_t> *tuning_result) -> cl_int {
    MACE_CHECK(params.size() == 4)
        << "Tuning parameters of 3D kernel must be 4D";
    cl_int error = CL_SUCCESS;
    std::vector<uint32_t> internal_gws(gws, gws + 3);
    if (!runtime->IsNonUniformWorkgroupsSupported()) {
      for (size_t i = 0; i < 3; ++i) {
        MACE_CHECK(params[i] != 0);
        internal_gws[i] = RoundUp(gws[i], params[i]);
      }
    }

    if (timer == nullptr) {
      uint32_t block_size = params[3] == 0 ? internal_gws[2] : params[3];
      const uint32_t num_blocks =
          RoundUpDiv<uint32_t>(internal_gws[2], block_size);
      for (uint32_t i = 0; i < num_blocks; ++i) {
        uint32_t gws2 = block_size;
        if (runtime->IsNonUniformWorkgroupsSupported() &&
            (i == num_blocks - 1)) {
          gws2 = (internal_gws[2] - (i * block_size));
        }
        error = runtime->command_queue().enqueueNDRangeKernel(
            kernel, cl::NDRange(0, 0, i * block_size),
            cl::NDRange(internal_gws[0], internal_gws[1], gws2),
            cl::NDRange(params[0], params[1], params[2]), nullptr, &event);
        MACE_CL_RET_ERROR(error);
      }
    } else {
      timer->ClearTiming();
      error = runtime->command_queue().enqueueNDRangeKernel(
          kernel, cl::NullRange,
          cl::NDRange(internal_gws[0], internal_gws[1], internal_gws[2]),
          cl::NDRange(params[0], params[1], params[2]), nullptr, &event);
      MACE_CL_RET_ERROR(error);
      timer->AccumulateTiming();
      tuning_result->assign(params.begin(), params.end());

      if (LimitKernelTime()) {
        double elapse_time = timer->AccumulatedMicros();
        timer->ClearTiming();
        uint32_t num_blocks = std::min(
            static_cast<uint32_t>(elapse_time / kMaxKernelExecTime) + 1,
            gws[2]);
        uint32_t block_size = gws[2] / num_blocks;
        if (!runtime->IsNonUniformWorkgroupsSupported()) {
          block_size = RoundUp(block_size, params[2]);
        }
        (*tuning_result)[3] = block_size;
        num_blocks = RoundUpDiv<uint32_t>(internal_gws[2], block_size);
        for (uint32_t i = 0; i < num_blocks; ++i) {
          uint32_t gws2 = block_size;
          if (runtime->IsNonUniformWorkgroupsSupported() &&
              (i == num_blocks - 1)) {
            gws2 = (internal_gws[2] - (i * block_size));
          }
          error = runtime->command_queue().enqueueNDRangeKernel(
              kernel, cl::NDRange(0, 0, i * block_size),
              cl::NDRange(internal_gws[0], internal_gws[1], gws2),
              cl::NDRange(params[0], params[1], params[2]), nullptr, &event);
          MACE_CL_RET_ERROR(error);
          timer->AccumulateTiming();
        }
      }
    }
    return error;
  };
  OpenCLProfilingTimer timer(&event);
  cl_int err = Tuner<uint32_t>::Get()->template TuneOrRun<cl_int>(
      tuning_key, lws, params_generator, func, &timer);
  MACE_CL_RET_STATUS(err);

  if (future != nullptr) {
    future->wait_fn = [event](CallStats *stats) {
      event.wait();
      if (stats != nullptr) {
        OpenCLRuntime::Global()->GetCallStats(event, stats);
      }
    };
  }
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus TuningOrRun2DKernel(const cl::Kernel &kernel,
                               const std::string tuning_key,
                               const uint32_t *gws,
                               const std::vector<uint32_t> &lws,
                               StatsFuture *future) {
  auto runtime = OpenCLRuntime::Global();

  auto params_generator = [&]() -> std::vector<std::vector<uint32_t>> {
    const uint32_t kwg_size =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel));
    std::vector<std::vector<uint32_t>> results;
    std::vector<std::vector<uint32_t>> candidates = {
        {kwg_size / 2, 2, 0},     {kwg_size / 4, 4, 0},
        {kwg_size / 8, 8, 0},     {kwg_size / 16, 16, 0},
        {kwg_size / 32, 32, 0},   {kwg_size / 64, 64, 0},
        {kwg_size / 128, 128, 0}, {kwg_size / 256, 256, 0},
        {kwg_size, 1, 0},         {1, kwg_size, 0}};
    for (auto &ele : candidates) {
      const uint32_t tmp = ele[0] * ele[1];
      if (0 < tmp && tmp <= kwg_size) {
        results.push_back(ele);
      }
    }
    return results;
  };
  cl::Event event;
  auto func = [&](const std::vector<uint32_t> &params, Timer *timer,
                  std::vector<uint32_t> *tuning_result) -> cl_int {
    MACE_CHECK(params.size() == 3)
        << "Tuning parameters of 2D kernel must be 3d";
    cl_int error = CL_SUCCESS;
    std::vector<uint32_t> internal_gws(gws, gws + 2);
    if (!runtime->IsNonUniformWorkgroupsSupported()) {
      for (size_t i = 0; i < 2; ++i) {
        MACE_CHECK(params[i] != 0);
        internal_gws[i] = RoundUp(gws[i], params[i]);
      }
    }

    if (timer == nullptr) {
      uint32_t block_size = params[2] == 0 ? internal_gws[1] : params[2];
      const uint32_t num_blocks =
          RoundUpDiv<uint32_t>(internal_gws[1], block_size);
      for (uint32_t i = 0; i < num_blocks; ++i) {
        uint32_t gws1 = block_size;
        if (runtime->IsNonUniformWorkgroupsSupported() &&
            (i == num_blocks - 1)) {
          gws1 = (internal_gws[1] - (i * block_size));
        }
        error = runtime->command_queue().enqueueNDRangeKernel(
            kernel, cl::NDRange(0, i * block_size),
            cl::NDRange(internal_gws[0], gws1),
            cl::NDRange(params[0], params[1]), nullptr, &event);
        MACE_CL_RET_ERROR(error);
      }
    } else {
      timer->ClearTiming();
      error = runtime->command_queue().enqueueNDRangeKernel(
          kernel, cl::NullRange, cl::NDRange(internal_gws[0], internal_gws[1]),
          cl::NDRange(params[0], params[1]), nullptr, &event);
      MACE_CL_RET_ERROR(error);
      timer->AccumulateTiming();
      tuning_result->assign(params.begin(), params.end());

      if (LimitKernelTime()) {
        double elapse_time = timer->AccumulatedMicros();
        timer->ClearTiming();
        uint32_t num_blocks = std::min(
            static_cast<uint32_t>(elapse_time / kMaxKernelExecTime) + 1,
            gws[1]);
        uint32_t block_size = gws[1] / num_blocks;
        if (!runtime->IsNonUniformWorkgroupsSupported()) {
          block_size = RoundUp(block_size, params[1]);
        }
        (*tuning_result)[2] = block_size;
        num_blocks = RoundUpDiv<uint32_t>(internal_gws[1], block_size);
        for (uint32_t i = 0; i < num_blocks; ++i) {
          uint32_t gws1 = block_size;
          if (runtime->IsNonUniformWorkgroupsSupported() &&
              (i == num_blocks - 1)) {
            gws1 = (internal_gws[1] - (i * block_size));
          }
          error = runtime->command_queue().enqueueNDRangeKernel(
              kernel, cl::NDRange(0, i * block_size),
              cl::NDRange(internal_gws[0], gws1),
              cl::NDRange(params[0], params[1]), nullptr, &event);
          MACE_CL_RET_ERROR(error);
          timer->AccumulateTiming();
        }
      }
    }
    return error;
  };
  OpenCLProfilingTimer timer(&event);
  cl_int err = Tuner<uint32_t>::Get()->template TuneOrRun<cl_int>(
      tuning_key, lws, params_generator, func, &timer);
  MACE_CL_RET_STATUS(err);

  if (future != nullptr) {
    future->wait_fn = [runtime, event](CallStats *stats) {
      event.wait();
      if (stats != nullptr) {
        runtime->GetCallStats(event, stats);
      }
    };
  }
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace kernels
}  // namespace mace
