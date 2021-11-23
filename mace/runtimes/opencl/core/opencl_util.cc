// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#include "mace/runtimes/opencl/core/opencl_util.h"

#include <utility>

#include "mace/core/ops/op_condition_context.h"
#include "mace/core/runtime/runtime.h"
#include "mace/utils/logging.h"
#include "mace/utils/math.h"

namespace mace {

namespace {
// [(C + 3) / 4 * W, N * H]
void CalInOutputImageShape(const std::vector<index_t> &shape, /* NHWC */
                           std::vector<size_t> *image_shape) {
  MACE_CHECK(shape.size() == 4, shape.size());
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
  auto shape_size = shape.size();
  if (shape_size > 1) {
    for (size_t i = 1; i < shape_size; ++i) {
      MACE_CHECK(shape[i] == 1, "The value of shape[1:] should be 1.");
    }
  } else {
    MACE_CHECK(shape.size() == 1, "The shape should be 1D.");
  }
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
  auto shape_size = shape.size();
  MACE_CHECK(shape_size == 4 || shape_size == 3);

  image_shape->resize(2);
  if (shape_size == 4) {
    (*image_shape)[0] = shape[2] * shape[3];
  } else if (shape_size == 3) {
    (*image_shape)[0] = shape[2];
  }
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

void OpenCLUtil::CalImage2DShape(const std::vector<index_t> &shape, /* NHWC */
                                 const BufferContentType type,
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
    case IN_OUT_CHANNEL: {
      const auto shape_size = shape.size();
      if (shape_size == 4) {
        CalInOutputImageShape(shape, image_shape);
      } else if (shape_size == 2) {
        CalInOutputImageShape({shape[0], 1, 1, shape[1]}, image_shape);
      } else if (shape_size == 1) {
        CalInOutputImageShape({1, 1, 1, shape[0]}, image_shape);
      } else if (shape_size == 0) {
        CalInOutputImageShape({1, 1, 1, 1}, image_shape);
      } else {
        MACE_NOT_IMPLEMENTED;
      }

      break;
    }
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

cl_channel_type OpenCLUtil::DataTypeToCLChannelType(const DataType t) {
  switch (t) {
    case DT_HALF:
      return CL_HALF_FLOAT;
    case DT_FLOAT:
      return CL_FLOAT;
    case DT_INT32:
      return CL_SIGNED_INT32;
    case DT_UINT8:
      return CL_UNSIGNED_INT32;
    default:
      LOG(FATAL) << "Image doesn't support the data type: " << t;
      return 0;
  }
}

void OpenCLUtil::SetOpenclInputToCpuBuffer(OpConditionContext *context,
                                           size_t idx, DataType data_type) {
  context->set_output_mem_type(context->runtime()->GetUsedMemoryType());

  if (context->operator_def()->input_size() > static_cast<int>(idx)) {
    context->SetInputInfo(idx, MemoryType::CPU_BUFFER, data_type);
  }
}

}  // namespace mace
