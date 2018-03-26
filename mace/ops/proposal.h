//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_PROPOSAL_H_
#define MACE_OPS_PROPOSAL_H_

#include "mace/core/operator.h"
#include "mace/kernels/proposal.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class ProposalOp : public Operator<D, T> {
 public:
  ProposalOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws),
        functor_(OperatorBase::GetSingleArgument<int>("min_size", 16),
                 OperatorBase::GetSingleArgument<float>("nms_thresh", 0.7),
                 OperatorBase::GetSingleArgument<int>("pre_nms_top_n", 6000),
                 OperatorBase::GetSingleArgument<int>("post_nms_top_n", 300),
                 OperatorBase::GetSingleArgument<int>("feat_stride", 0),
                 OperatorBase::GetSingleArgument<int>("base_size", 12),
                 OperatorBase::GetRepeatedArgument<int>("scales"),
                 OperatorBase::GetRepeatedArgument<float>("ratios")) {}

  bool Run(StatsFuture *future) override {
    const Tensor *rpn_cls_prob = this->Input(RPN_CLS_PROB);
    const Tensor *rpn_bbox_pred = this->Input(RPN_BBOX_PRED);
    const Tensor *img_info = this->Input(IMG_INFO);

    Tensor *output = this->Output(ROIS);

    functor_(rpn_cls_prob, rpn_bbox_pred, img_info, output, future);
    return true;
  }

 private:
  kernels::ProposalFunctor<D, T> functor_;

 protected:
  OP_INPUT_TAGS(RPN_CLS_PROB, RPN_BBOX_PRED, IMG_INFO);
  OP_OUTPUT_TAGS(ROIS);
};

}  // namespace ops
}  // namespace mace

#endif  //  MACE_OPS_PROPOSAL_H_
