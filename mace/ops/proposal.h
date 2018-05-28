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
        functor_(OperatorBase::GetOptionalArg<int>("min_size", 16),
                 OperatorBase::GetOptionalArg<float>("nms_thresh", 0.7),
                 OperatorBase::GetOptionalArg<int>("pre_nms_top_n", 6000),
                 OperatorBase::GetOptionalArg<int>("post_nms_top_n", 300),
                 OperatorBase::GetOptionalArg<int>("feat_stride", 0),
                 OperatorBase::GetOptionalArg<int>("base_size", 12),
                 OperatorBase::GetRepeatedArgs<int>("scales"),
                 OperatorBase::GetRepeatedArgs<float>("ratios")) {}

  MaceStatus Run(StatsFuture *future) override {
    const Tensor *rpn_cls_prob = this->Input(RPN_CLS_PROB);
    const Tensor *rpn_bbox_pred = this->Input(RPN_BBOX_PRED);
    const Tensor *img_info = this->Input(IMG_INFO);

    Tensor *output = this->Output(ROIS);

    return functor_(rpn_cls_prob, rpn_bbox_pred, img_info, output, future);
  }

 private:
  kernels::ProposalFunctor<D, T> functor_;

 protected:
  MACE_OP_INPUT_TAGS(RPN_CLS_PROB, RPN_BBOX_PRED, IMG_INFO);
  MACE_OP_OUTPUT_TAGS(ROIS);
};

}  // namespace ops
}  // namespace mace

#endif  //  MACE_OPS_PROPOSAL_H_
