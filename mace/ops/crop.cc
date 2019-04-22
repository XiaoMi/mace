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

#include <memory>

#include "mace/core/operator.h"
#include "mace/utils/math.h"
#include "mace/utils/memory.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/image/crop.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace ops {

template <DeviceType D, class T>
class CropOp;

template <class T>
class CropOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit CropOp(OpConstructContext *context)
      : Operation(context),
        offset_(Operation::GetRepeatedArgs<int>("offset")) {
    MACE_CHECK(offset_.size() == 4,
               "crop op only supports 4-dims inputs now.");
    auto has_df = Operation::GetOptionalArg<int>(
        "has_data_format", 0);
    if (has_df) {
      // NHWC -> NCHW
      offset_ = TransposeShape<int, int>(offset_, {0, 3, 1, 2});
    }
  }


  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    MACE_CHECK(inputs_.size() == 2, "Crop op needs two inputs.");
    Tensor *output = this->Output(0);
    const Tensor *input0 = inputs_[0];
    const Tensor *input1 = inputs_[1];
    const uint32_t in0_dims = static_cast<uint32_t >(input0->dim_size());
    const uint32_t in1_dims = static_cast<uint32_t >(input0->dim_size());

    MACE_CHECK(in0_dims == 4 && in1_dims == 4,
               "crop op only supports 4-dims inputs now.");

    std::vector<int32_t> offsets(in0_dims, 0);

    std::vector<index_t> output_shape(input0->shape());
    for (index_t i = 0; i < in0_dims; ++i) {
      if (offset_[i] >= 0) {
        output_shape[i] = input1->dim(i);
        offsets[i] = offset_[i];
        MACE_CHECK(input0->dim(i) - offset_[i] >= input1->dim(i))
          << "the crop for dimension " << i << " is out of bound with size "
          << input1->dim(i) << " and offset " << offsets[i];
      }
    }
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));
    T *output_data = output->mutable_data<T>();

    const T * input_data = input0->data<T>();

    crop_copy(input_data, output_data, input0->shape(),
              output_shape, offsets.data());

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  void crop_copy(const T* input_data, T* output_data,
                 const std::vector<index_t> &input_shape,
                 const std::vector<index_t> &output_shape,
                 const int32_t* offsets) {
    const index_t out_img_size =
        output_shape[1] * output_shape[2] * output_shape[3];
    const index_t out_hw = output_shape[2] * output_shape[3];
    const index_t in_img_size =
        input_shape[1] * input_shape[2] * input_shape[3];
    const index_t in_hw = input_shape[2] * input_shape[3];

    for (int b = 0; b < output_shape[0]; ++b) {
      for (int c = 0; c < output_shape[1]; ++c) {
        for (int h = 0; h < output_shape[2]; ++h) {
          T* out_ptr =
              output_data + b * out_img_size + c * out_hw + h * output_shape[3];
          const T* in_ptr_bch =
              input_data + (b + offsets[0]) * in_img_size +
                  (c + offsets[1]) * in_hw +
                  (h + offsets[2]) * input_shape[3] + offsets[3];
          memcpy(out_ptr, in_ptr_bch,
                 output_shape[3] * sizeof(T));
        }
      }
    }
  }

 private:
  std::vector<int> offset_;
};

#ifdef MACE_ENABLE_OPENCL
template <typename T>
class CropOp<DeviceType::GPU, T> : public Operation {
 public:
  explicit CropOp(OpConstructContext *context)
      : Operation(context) {
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::CropKernel<T>>(
          Operation::GetRepeatedArgs<int>("offset"));
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }
  MaceStatus Run(OpContext *context) override {
    return kernel_->Compute(context, inputs_, this->Output(0));
  }

 private:
  std::unique_ptr<OpenCLCropKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL


void RegisterCrop(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "Crop", CropOp,
                   DeviceType::CPU, float);

#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_OP(op_registry, "Crop", CropOp,
                   DeviceType::GPU, float);

  MACE_REGISTER_OP(op_registry, "Crop", CropOp,
                   DeviceType::GPU, half);
#endif  // MACE_ENABLE_OPENCL
  MACE_REGISTER_OP_CONDITION(
      op_registry,
      OpConditionBuilder("Crop")
          .SetDevicePlacerFunc(
              [](OpConditionContext *context) -> std::set<DeviceType> {
                auto op = context->operator_def();
                if (op->output_shape_size() != op->output_size()) {
                  return { DeviceType::CPU, DeviceType::GPU };
                }
                int has_data_format =
                    ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
                        *op, "has_data_format", 0);
                if (!has_data_format ||
                    op->output_shape(0).dims_size() != 4) {
                  return { DeviceType::CPU };
                }
                return { DeviceType::CPU, DeviceType::GPU };
              }));
}

}  // namespace ops
}  // namespace mace
