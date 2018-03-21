//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_PROPOSAL_H_
#define MACE_KERNELS_PROPOSAL_H_

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/core/public/mace.h"

namespace mace {
namespace kernels {

static std::vector<float> WHCenters(const std::vector<float> &anchor) {
  // width, height, width_center, height_center
  std::vector<float> window(4);
  window[0] = anchor[2] - anchor[0] + 1;
  window[1] = anchor[3] - anchor[1] + 1;
  window[2] = anchor[0] + (window[0] - 1) / 2;
  window[3] = anchor[1] + (window[1] - 1) / 2;
  return window;
}

std::vector<std::vector<float>> GenerateAnchors(const std::vector<int> &scales, const std::vector<float> &ratios,
                     const int base_size = 16) {
  const std::vector<float> base_anchor = {0, 0, (float)base_size-1, (float)base_size-1};

  const size_t scales_size = scales.size();
  const size_t ratios_size = ratios.size();
  // get height, width, centers
  std::vector<float> base_window = WHCenters(base_anchor);
  const float size = base_window[0] * base_window[1];
  std::vector<std::vector<float>> anchors(scales_size * ratios_size, std::vector<float>(4));

  int idx = 0;
  std::vector<float> tmp_anchor(4);
  for (size_t ratio_idx = 0; ratio_idx < ratios_size; ++ratio_idx) {
    float ws = ::roundf(::sqrtf(size / ratios[ratio_idx]));
    float hs = ::roundf(ws * ratios[ratio_idx]);
    tmp_anchor[0] = base_window[2] - (ws - 1) / 2;
    tmp_anchor[1] = base_window[3] - (hs - 1) / 2;
    tmp_anchor[2] = base_window[2] + (ws - 1) / 2;
    tmp_anchor[3] = base_window[3] + (hs - 1) / 2;
    auto window = WHCenters(tmp_anchor);
    for (size_t scale_idx = 0; scale_idx < scales_size; ++scale_idx) {
      ws = window[0] * scales[scale_idx];
      hs = window[1] * scales[scale_idx];
      anchors[idx][0] = window[2] - (ws - 1) / 2;
      anchors[idx][1] = window[3] - (hs - 1) / 2;
      anchors[idx][2] = window[2] + (ws - 1) / 2;
      anchors[idx][3] = window[3] + (hs - 1) / 2;
      idx++;
    }
  }
  return anchors;
}


template<DeviceType D, typename T>
struct ProposalFunctor {
  ProposalFunctor(const int feat_stride, const std::vector<int> &scales) :
      feat_stride_(feat_stride),
      scales_(scales),
      anchors_(GenerateAnchors(scales, {0.5, 1, 2})) {}

  void operator()(const Tensor *rpn_cls_prob,
                  const Tensor *rpn_bbox_pred,
                  const Tensor *im_info,
                  Tensor *output,
                  StatsFuture *future) {
    const index_t feat_height = rpn_cls_prob->dim(1);
    const index_t feat_width = rpn_cls_prob->dim(2);
    const int anchors_size = anchors_.size();

    // shift anchors
    std::vector<std::vector<float>> shifted_anchors(anchors_.size() * feat_height * feat_width,
                                                    std::vector<float>(4));
    int shift_w, shift_h;
    int sanc_idx = 0;
    for (int h_idx = 0; h_idx < feat_height; ++h_idx) {
      shift_h = h_idx * feat_stride_;
      for (int w_idx = 0; w_idx < feat_width; ++w_idx) {
        shift_w = w_idx * feat_stride_;
        for (int a_idx = 0; a_idx < anchors_size; ++a_idx) {
          shifted_anchors[sanc_idx][0] = anchors_[a_idx][0] + shift_w;
          shifted_anchors[sanc_idx][1] = anchors_[a_idx][1] + shift_h;
          shifted_anchors[sanc_idx][2] = anchors_[a_idx][2] + shift_w;
          shifted_anchors[sanc_idx][3] = anchors_[a_idx][3] + shift_h;
          sanc_idx++;
        }
      }
    }
    // Convert anchors into proposals via bbox transformations

    // clip predicted boxes to image

    // remove predicted boxes with either height or width < threshold

    // 4. sort all (proposal, score) pairs by score from highest to lowest
    // 5. take top pre_nms_topN (e.g. 6000)

    /* 6. apply nms (e.g. threshold = 0.7)
       7. take after_nms_topN (e.g. 300)
       8. return the top proposals (-> RoIs top) */

    // Output rois blob
    // Our RPN implementation only supports a single input image, so all
    // batch inds are 0
  }

  const int feat_stride_;
  const std::vector<int> scales_;
  std::vector<std::vector<float>> anchors_;
};

}  //  namepsace kernels
}  //  namespace mace

#endif  //  MACE_KERNELS_PROPOSAL_H_
