//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_PROPOSAL_H_
#define MACE_KERNELS_PROPOSAL_H_

#include <algorithm>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/public/mace.h"

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

std::vector<std::vector<float>> GenerateAnchors(
    const std::vector<int> &scales,
    const std::vector<float> &ratios,
    const int base_size) {
  const std::vector<float> base_anchor =
      {0, 0,
       static_cast<float>(base_size-1),
       static_cast<float>(base_size-1)};

  const size_t scales_size = scales.size();
  const size_t ratios_size = ratios.size();
  // get height, width, centers
  std::vector<float> base_window = WHCenters(base_anchor);
  const float size = base_window[0] * base_window[1];
  std::vector<std::vector<float>> anchors(scales_size * ratios_size,
                                          std::vector<float>(4));

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

std::vector<int> nms(const float *bboxes_ptr,
                     const float *scores_ptr,
                     const index_t num_bboxes,
                     const float thresh) {
  std::vector<int> keep;
  std::vector<int> suppressed(num_bboxes, 0);

  std::vector<float> areas(num_bboxes, 0);
  for (index_t i = 0; i < num_bboxes; ++i) {
    const index_t idx = (i << 2);
    areas[i] = (bboxes_ptr[idx + 2] - bboxes_ptr[idx] + 1) *
        (bboxes_ptr[idx + 3] - bboxes_ptr[idx + 1] + 1);
  }

  std::vector<int> order(num_bboxes, 0);
  iota(order.begin(), order.end(), 0);

  float x1, y1, x2, y2, area1, iou;
  for (index_t i = 0; i < num_bboxes; ++i) {
    int idx = order[i];
    if (suppressed[idx] == 1) continue;
    keep.push_back(idx);
    int coord_idx = idx << 2;
    x1 = bboxes_ptr[coord_idx];
    y1 = bboxes_ptr[coord_idx + 1];
    x2 = bboxes_ptr[coord_idx + 2];
    y2 = bboxes_ptr[coord_idx + 3];
    area1 = areas[idx];
    for (index_t j = i + 1; j < num_bboxes; ++j) {
      const int other_idx = order[j];
      if (suppressed[other_idx] == 1) continue;

      coord_idx = other_idx << 2;
      iou = std::max<float>(0.0,
                       std::min(x2, bboxes_ptr[coord_idx + 2]) -
                       std::max(x1, bboxes_ptr[coord_idx]) + 1)
          * std::max<float>(0.0,
                       std::min(y2, bboxes_ptr[coord_idx + 3]) -
                       std::max(y1, bboxes_ptr[coord_idx + 1]) + 1);
      if ((iou / (area1 + areas[other_idx] - iou)) >= thresh) {
        suppressed[other_idx] = 1;
      }
    }
  }
  return keep;
}


template<DeviceType D, typename T>
struct ProposalFunctor {
  ProposalFunctor(const int min_size,
                  const float nms_thresh,
                  const int pre_nms_top_n,
                  const int post_nms_top_n,
                  const int feat_stride,
                  const int base_size,
                  const std::vector<int> &scales,
                  const std::vector<float> &ratios) :
      min_size_(min_size),
      thresh_(nms_thresh),
      pre_nms_top_n_(pre_nms_top_n),
      post_nms_top_n_(post_nms_top_n),
      feat_stride_(feat_stride),
      anchors_(GenerateAnchors(scales, ratios, base_size)) {}

  void operator()(const Tensor *rpn_cls_prob,
                  const Tensor *rpn_bbox_pred,
                  const Tensor *img_info_tensor,
                  Tensor *output,
                  StatsFuture *future) {
    MACE_CHECK(rpn_cls_prob->dim(1) == rpn_bbox_pred->dim(1) &&
        rpn_cls_prob->dim(2) == rpn_bbox_pred->dim(2));
    MACE_CHECK((rpn_cls_prob->dim(3) / 2 == rpn_bbox_pred->dim(3) / 4) &&
        (rpn_cls_prob->dim(3) / 2 == anchors_.size()));
    const float *img_info = img_info_tensor->data<float>();
    const index_t im_height = img_info[0] - 1;
    const index_t im_width = img_info[1] - 1;
    const index_t feat_height = rpn_cls_prob->dim(1);
    const index_t feat_width = rpn_cls_prob->dim(2);
    const int anchors_size = anchors_.size();

    // shift anchors to original input
    std::vector<std::vector<float>> proposals(
        anchors_size * feat_height * feat_width,
        std::vector<float>(4));
    int shift_w, shift_h;
    int sanc_idx = 0;
    for (int h_idx = 0; h_idx < feat_height; ++h_idx) {
      shift_h = h_idx * feat_stride_;
      for (int w_idx = 0; w_idx < feat_width; ++w_idx) {
        shift_w = w_idx * feat_stride_;
        for (int a_idx = 0; a_idx < anchors_size; ++a_idx) {
          proposals[sanc_idx][0] = anchors_[a_idx][0] + shift_w;
          proposals[sanc_idx][1] = anchors_[a_idx][1] + shift_h;
          proposals[sanc_idx][2] = anchors_[a_idx][2] + shift_w;
          proposals[sanc_idx][3] = anchors_[a_idx][3] + shift_h;
          sanc_idx++;
        }
      }
    }
    // Convert anchors into proposals via bbox transformations
    // 2. clip predicted boxes to image
    std::vector<int> keep;
    const index_t min_size = min_size_ * img_info[2];
    // 3. remove predicted boxes with either height or width < threshold
    // (NOTE: convert min_size to input image scale stored in im_info[2])

    const float *bbox_deltas = rpn_bbox_pred->data<float>();
    sanc_idx = 0;
    for (int h_idx = 0; h_idx < feat_height; ++h_idx) {
      for (int w_idx = 0; w_idx < feat_width; ++w_idx) {
        for (int a_idx = 0; a_idx < anchors_size; ++a_idx) {
          float width = proposals[sanc_idx][2] - proposals[sanc_idx][0] + 1;
          float height = proposals[sanc_idx][3] - proposals[sanc_idx][1] + 1;
          int delta_offset = sanc_idx * 4;
          float pred_ctr_x = bbox_deltas[delta_offset + 0] * width +
              (proposals[sanc_idx][0] + width / 2);
          float pred_ctr_y = bbox_deltas[delta_offset + 1] * height +
              (proposals[sanc_idx][1] + height / 2);
          float pred_w = std::exp(bbox_deltas[delta_offset + 2]) * width;
          float pred_h = std::exp(bbox_deltas[delta_offset + 3]) * height;

          proposals[sanc_idx][0] = std::max<float>(
              std::min<float>(pred_ctr_x - pred_w / 2, im_width),
              0);
          proposals[sanc_idx][1] = std::max<float>(
              std::min<float>(pred_ctr_y - pred_h / 2, im_height),
              0);
          proposals[sanc_idx][2] = std::max<float>(
              std::min<float>(pred_ctr_x + pred_w / 2, im_width),
              0);
          proposals[sanc_idx][3] = std::max<float>(
              std::min<float>(pred_ctr_y + pred_h / 2, im_height),
              0);

          width = proposals[sanc_idx][2] - proposals[sanc_idx][0] + 1;
          height = proposals[sanc_idx][3] - proposals[sanc_idx][1] + 1;
          if (width >= min_size && height >= min_size) {
            keep.push_back(sanc_idx);
          }
          sanc_idx++;
        }
      }
    }

    // 4. sort all (proposal, score) pairs by score from highest to lowest
    // 5. take top pre_nms_topN (e.g. 6000)
    auto scores = rpn_cls_prob->data<float>();
    const int scores_chan = static_cast<int>(rpn_cls_prob->dim(3));

    auto score_idx_func = [&](int idx) -> int {
      return (idx / anchors_size) * scores_chan +
          (idx % anchors_size) + anchors_size;
    };
    std::sort(keep.begin(), keep.end(), [&](int left, int right) -> bool{
      return scores[score_idx_func(left)] >
          scores[score_idx_func(right)];
    });

    int size = std::min<int>(pre_nms_top_n_, keep.size());
    std::vector<float> nms_scores(size, 0);
    std::vector<float> nms_proposals((size << 2), 0);
    for (int i = 0; i < size; ++i) {
      nms_scores[i] = scores[score_idx_func(keep[i])];
      nms_proposals[i << 2] = proposals[keep[i]][0];
      nms_proposals[(i << 2) + 1] = proposals[keep[i]][1];
      nms_proposals[(i << 2) + 2] = proposals[keep[i]][2];
      nms_proposals[(i << 2) + 3] = proposals[keep[i]][3];
    }

    /* 6. apply nms (e.g. threshold = 0.7)
       7. take after_nms_topN (e.g. 300)
       8. return the top proposals (-> RoIs top) */
    auto nms_result = nms(nms_proposals.data(), nms_scores.data(),
                          nms_scores.size(), thresh_);

    // Output rois blob
    // Our RPN implementation only supports a single input image, so all
    // batch inds are 0
    size = std::min<int>(post_nms_top_n_, nms_result.size());
    output->Resize({1, 1, size, 5});
    auto output_ptr = output->mutable_data<float>();
    for (int i = 0; i < size; ++i) {
      const int out_idx = i * 5;
      const int nms_idx = nms_result[i] * 4;
      output_ptr[out_idx] = 0;
      output_ptr[out_idx + 1] = nms_proposals[nms_idx];
      output_ptr[out_idx + 2] = nms_proposals[nms_idx + 1];
      output_ptr[out_idx + 3] = nms_proposals[nms_idx + 2];
      output_ptr[out_idx + 4] = nms_proposals[nms_idx + 3];
    }
  }

  const int min_size_;
  const float thresh_;
  const int pre_nms_top_n_;
  const int post_nms_top_n_;
  const int feat_stride_;
  std::vector<std::vector<float>> anchors_;
};

}  // namespace kernels
}  // namespace mace

#endif  //  MACE_KERNELS_PROPOSAL_H_
