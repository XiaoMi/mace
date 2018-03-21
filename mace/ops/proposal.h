//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_SOFTMAX_H_
#define MACE_SOFTMAX_H_

#include "mace/core/operator.h"
#include "mace/kernels/proposal.h"

namespace mace {

template <DeviceType D, class T>
class ProposalOp : public Operator<D, T> {
 public:
  ProposalOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws),
        functor_(OperatorBase::GetSingleArgument<int>("feat_stride", 1),
                 OperatorBase::GetRepeatedArgument<int>("scales")) {}

  bool Run(StatsFuture *future) override {
    const Tensor *rpn_cls_prob = this->Input(RPN_CLS_PROB);
    const Tensor *rpn_bbox_pred = this->Input(RPN_BBOX_PRED);
    const Tensor *im_info = this->Input(IM_INFO);

    Tensor *output = this->Output(ROIS);

    functor_(rpn_cls_prob, rpn_bbox_pred, im_info, output, future);
    return true;
  }

 private:
  kernels::ProposalFunctor<D, T> functor_;

 protected:
  OP_INPUT_TAGS(RPN_CLS_PROB, RPN_BBOX_PRED, IM_INFO);
  OP_OUTPUT_TAGS(ROIS);
};

}  //  namespace mace

#endif  //  MACE_SOFTMAX_H_
