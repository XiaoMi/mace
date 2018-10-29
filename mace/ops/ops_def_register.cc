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

#include "mace/ops/ops_def_register.h"

#include <vector>

namespace mace {
namespace ops {

void RegisterOpDefs(OpDefRegistryBase *op_def_registry) {
  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("Activation")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("AddN")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("ArgMax")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("BatchNorm")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("BatchToSpaceND")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("BiasAdd")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("BufferInverseTransform")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("BufferTransform")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("Cast")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("ChannelShuffle")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("Concat")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("Conv2D")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("Crop")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("Deconv2D")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("DepthToSpace")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("DepthwiseConv2d")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("Dequantize")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("Eltwise")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("ExpandDims")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("Fill")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("FullyConnected")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("Gather")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("Identity")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("InferConv2dShape")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("LocalResponseNorm")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("LSTMCell")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("MatMul")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("Pad")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("Pooling")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("Quantize")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("ReduceMean")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("Reshape")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("ResizeBicubic")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("ResizeBilinear")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("Reverse")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("ScalarMath")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("Shape")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("Softmax")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("SpaceToBatchND")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("SpaceToDepth")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("Split")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("SqrDiffMean")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("Squeeze")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("Stack")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("StridedSlice")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("Transpose")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("Unstack")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("WinogradInverseTransform")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::GPU};
          }));

  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("WinogradTransform")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::GPU};
          }));
}
}  // namespace ops


OpDefRegistry::OpDefRegistry() : OpDefRegistryBase() {
  ops::RegisterOpDefs(this);
}

}  // namespace mace
