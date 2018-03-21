//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_SPACE_TO_DEPTH_H_
#define MACE_OPS_SPACE_TO_DEPTH_H_

#include <memory>
#include <vector>

#include "mace/core/operator.h"
#include "mace/kernels/space_to_depth.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class SpaceToDepthOp : public Operator<D, T> {
  public:
  SpaceToDepthOp(const OperatorDef &op_def, Workspace *ws)
      : Operator<D, T>(op_def, ws),
        functor_(OperatorBase::GetSingleArgument<int>("block_size", 1)) {}

  bool Run(StatsFuture *future) override {
	  const Tensor *input = this->Input(INPUT);
	  Tensor *output = this->Output(OUTPUT);
	  MACE_CHECK(input->dim_size() == 4, "input dim should be 4");
	  
	  const int block_size = OperatorBase::GetSingleArgument<int>("block_size", 1);

      const int input_height = input->dim(1);
      const int input_width  = input->dim(2);
	  const int input_depth = input->dim(3);
	  MACE_CHECK((input_width % block_size == 0) && (input_height % block_size == 0),
				 "input width and height should be dividable by block_size",
				 input->dim(3));
	  functor_(input, output, future);
	  return true;
  }
  
  protected:
    OP_INPUT_TAGS(INPUT);
    OP_OUTPUT_TAGS(OUTPUT);
    
  private:
    kernels::SpaceToDepthOpFunctor<D, T> functor_;

};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_SPACE_TO_DEPTH_H_
