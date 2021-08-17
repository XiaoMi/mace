// Copyright 2021 The MACE Authors. All Rights Reserved.
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

#include "mace/runtimes/qnn/op_builder.h"

#include "mace/core/proto/arg_helper.h"
#include "mace/ops/common/coordinate_transformation_mode.h"

namespace mace {
class ResizeOpBuilder : public OpBuilder {
 public:
  explicit ResizeOpBuilder(GraphBuilder *graph_builder)
      : OpBuilder(graph_builder) {
    names_ = {
        {"ResizeBilinear",
         {QNN_OP_RESIZE_BILINEAR, QNN_OP_RESIZE_BILINEAR_PARAM_ALIGN_CORNERS,
          QNN_OP_RESIZE_BILINEAR_PARAM_HALF_PIXEL_CENTERS}},
        {"ResizeNearestNeighbor",
         {QNN_OP_RESIZE_NEAREST_NEIGHBOR,
          QNN_OP_RESIZE_NEAREST_NEIGHBOR_PARAM_ALIGN_CORNERS,
          QNN_OP_RESIZE_NEAREST_NEIGHBOR_PARAM_HALF_PIXEL_CENTERS}}};
  }

  MaceStatus BuildOp(const OperatorDef &op, DataType quantized_type) {
    MACE_UNUSED(quantized_type);
    MACE_CHECK(names_.count(op.type()) > 0,
               "QNN does not support op: ", op.type());
    auto names = names_.at(op.type());
    SetOpType(names.op_type);
    SetOpName(op.name().c_str());

    int align_corners = ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
        op, "align_corners", 0);
    AddScalarParam(names.align_corners,
                   {QNN_DATATYPE_BOOL_8,
                    .bool8Value = static_cast<uint8_t>(align_corners)});

    auto coordinate_transformation_mode =
        static_cast<ops::CoordinateTransformationMode>(
            ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
                op, "coordinate_transformation_mode", 0));
    int half_pixel_centers =
        (coordinate_transformation_mode ==
                 ops::CoordinateTransformationMode::HALF_PIXEL
             ? 1
             : 0);
    if (coordinate_transformation_mode ==
        ops::CoordinateTransformationMode::PYTORCH_HALF_PIXEL) {
      index_t output_height = graph_builder_->GetTensorShape(op.output(0))[1];
      index_t output_width = graph_builder_->GetTensorShape(op.output(0))[2];
      MACE_CHECK(output_height > 1 && output_width > 1,
                 "QNN does not support pytorch_half_pixel when output_height "
                 "or output width is 0.");
      half_pixel_centers = 1;
    }
    AddScalarParam(names.half_pixel_centers,
                   {QNN_DATATYPE_BOOL_8,
                    .bool8Value = static_cast<uint8_t>(half_pixel_centers)});

    // Input1 is ignored as output shape is set already
    AddInput(op.input(0));
    AddOutput(op.output(0));

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  struct Names {
    const char *op_type;
    const char *align_corners;
    const char *half_pixel_centers;
  };
  std::unordered_map<std::string, Names> names_;
};
namespace qnn {
void RegisterResize(OpRegistry *op_registry) {
  QNN_REGISTER_OP(op_registry, "ResizeBilinear", ResizeOpBuilder);
  QNN_REGISTER_OP(op_registry, "ResizeNearestNeighbor", ResizeOpBuilder);
}
}  // namespace qnn
}  // namespace mace
