//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_DEPTH_TO_SPACE_H_
#define MACE_OPS_DEPTH_TO_SPACE_H_

#include <memory>
#include <vector>

#include "mace/core/operator.h"
#include "mace/kernels/depth_to_space.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class DepthToSpaceOp : public Operator<D, T> {
  public:
  DepthToSpaceOp(const OperatorDef &op_def, Workspace *ws)
      : Operator<D, T>(op_def, ws),
        block_size_(OperatorBase::GetSingleArgument<int>("block_size", 1)),
        functor_(this->block_size_) {}

  bool Run(StatsFuture *future) override {
	  const Tensor *input = this->Input(INPUT);
	  Tensor *output = this->Output(OUTPUT);
	  MACE_CHECK(input->dim_size() == 4, "input dim should be 4");

	  int input_depth = input->dim(3);
	  MACE_CHECK(input_depth % (block_size_ * block_size_) == 0,
				 "input depth should be dividable by block_size * block_size",
				 input->dim(3));

	  functor_(input, output, future);
	  return true;
  }

  private:
    kernels::DepthToSpaceOpFunctor<D, T> functor_;

  protected:
    const int block_size_;
    OP_INPUT_TAGS(INPUT);
    OP_OUTPUT_TAGS(OUTPUT);

};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_DEPTH_TO_SPACE_H_
