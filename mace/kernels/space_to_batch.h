//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_CONV_2D_H_
#define MACE_KERNELS_CONV_2D_H_

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/core/public/mace.h"
#include "mace/core/runtime/opencl/cl2_header.h"

namespace mace {
namespace kernels {

struct SpaceToBatchFunctorBase {
  SpaceToBatchFunctorBase(const std::vector<int> &paddings,
                          const std::vector<int> &block_shape,
                          bool b2s):
      paddings_(paddings.begin(), paddings.end()),
      block_shape_(block_shape.begin(), block_shape.end()),
      b2s_(b2s)
  {}

  std::vector<int> paddings_;
  std::vector<int> block_shape_;
  bool b2s_;
};

template <DeviceType D, typename T>
struct SpaceToBatchFunctor : SpaceToBatchFunctorBase{
  SpaceToBatchFunctor(const std::vector<int> &paddings,
                      const std::vector<int> &block_shape,
                      bool b2s): SpaceToBatchFunctorBase(paddings, block_shape, b2s){}

  void operator()(Tensor *space_tensor,
                  const std::vector<index_t> &output_shape,
                  Tensor *batch_tensor,
                  StatsFuture *future) {
    MACE_NOT_IMPLEMENTED;
  }
};

template <typename T>
struct SpaceToBatchFunctor<DeviceType::OPENCL, T>: SpaceToBatchFunctorBase{
  SpaceToBatchFunctor(const std::vector<int> &paddings,
                      const std::vector<int> &block_shape,
                      bool b2s): SpaceToBatchFunctorBase(paddings, block_shape, b2s){}

  void operator()(Tensor *space_tensor,
                  const std::vector<index_t> &output_shape,
                  Tensor *batch_tensor,
                  StatsFuture *future);

  cl::Kernel kernel_;

};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_CONV_2D_H_
