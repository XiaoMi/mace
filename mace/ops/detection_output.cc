// Copyright 2018 The MACE Authors. All Rights Reserved.
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


#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "mace/core/ops/operator.h"
#include "mace/core/quantize.h"
#include "mace/core/registry/ops_registry.h"
#include "mace/utils/logging.h"
#include "mace/utils/memory.h"


namespace mace {

struct BBox {
  float xmin;
  float ymin;
  float xmax;
  float ymax;
  int label;
  float confidence;
};

namespace {
inline float overlap(const BBox &a, const BBox &b) {
  if (a.xmin > b.xmax || a.xmax < b.xmin || a.ymin > b.ymax ||
      a.ymax < b.ymin) {
    return 0.f;
  }
  float overlap_w = std::min(a.xmax, b.xmax) - std::max(a.xmin, b.xmin);
  float overlap_h = std::min(a.ymax, b.ymax) - std::max(a.ymin, b.ymin);
  return overlap_w * overlap_h;
}

void NmsSortedBboxes(const std::vector<BBox> &bboxes,
                     const float nms_threshold,
                     const int top_k,
                     std::vector<BBox> *sorted_boxes) {
  const int n = std::min(top_k, static_cast<int>(bboxes.size()));
  std::vector<int> picked;

  std::vector<float> areas(n);
  for (int i = 0; i < n; ++i) {
    const BBox &r = bboxes[i];
    float width = std::max(0.f, r.xmax - r.xmin);
    float height = std::max(0.f, r.ymax - r.ymin);
    areas[i] = width * height;
  }

  for (int i = 0; i < n; ++i) {
    const BBox &a = bboxes[i];
    int keep = 1;
    for (size_t j = 0; j < picked.size(); ++j) {
      const BBox &b = bboxes[picked[j]];

      float inter_area = overlap(a, b);
      float union_area = areas[i] + areas[picked[j]] - inter_area;
      MACE_CHECK(union_area > 0, "union_area should be greater than 0");
      if (inter_area / union_area > nms_threshold) {
        keep = 0;
        break;
      }
    }

    if (keep) {
      picked.push_back(i);
      sorted_boxes->push_back(bboxes[i]);
    }
  }
}

inline bool cmp(const BBox &a, const BBox &b) {
  return a.confidence > b.confidence;
}
}  // namespace

int DetectionOutput_CLA(const float *loc_ptr,
                        const float *conf_ptr,
                        const float *pbox_ptr,
                        const int num_prior,
                        const int num_classes,
                        const float nms_threshold,
                        const int top_k,
                        const int keep_top_k,
                        const float confidence_threshold,
                        std::vector<BBox> *bbox_rects) {
  MACE_CHECK(keep_top_k > 0, "keep_top_k should be greater than 0");
  std::vector<float> bboxes(4 * num_prior);
  for (int i = 0; i < num_prior; ++i) {
    int index = i * 4;
    const float *lc = loc_ptr + index;
    const float *pb = pbox_ptr + index;
    const float *var = pb + num_prior * 4;

    float pb_w = pb[2] - pb[0];
    float pb_h = pb[3] - pb[1];
    float pb_cx = (pb[0] + pb[2]) * 0.5f;
    float pb_cy = (pb[1] + pb[3]) * 0.5f;

    float bbox_cx = var[0] * lc[0] * pb_w + pb_cx;
    float bbox_cy = var[1] * lc[1] * pb_h + pb_cy;
    float bbox_w = std::exp(var[2] * lc[2]) * pb_w;
    float bbox_h = std::exp(var[3] * lc[3]) * pb_h;

    bboxes[0 + index] = bbox_cx - bbox_w * 0.5f;
    bboxes[1 + index] = bbox_cy - bbox_h * 0.5f;
    bboxes[2 + index] = bbox_cx + bbox_w * 0.5f;
    bboxes[3 + index] = bbox_cy + bbox_h * 0.5f;
  }
  // start from 1 to ignore background class

  for (int i = 1; i < num_classes; ++i) {
    // filter by confidence threshold
    std::vector<BBox> class_bbox_rects;
    for (int j = 0; j < num_prior; ++j) {
      float confidence = conf_ptr[j * num_classes + i];
      if (confidence > confidence_threshold) {
        BBox c = {bboxes[0 + j * 4],
                  bboxes[1 + j * 4],
                  bboxes[2 + j * 4],
                  bboxes[3 + j * 4],
                  i,
                  confidence};
        class_bbox_rects.push_back(c);
      }
    }
    std::sort(class_bbox_rects.begin(), class_bbox_rects.end(), cmp);

    // apply nms
    std::vector<BBox> sorted_boxes;
    NmsSortedBboxes(class_bbox_rects, nms_threshold,
                    std::min(top_k, static_cast<int>(class_bbox_rects.size())),
                    &sorted_boxes);
    // gather
    bbox_rects->insert(bbox_rects->end(), sorted_boxes.begin(),
                       sorted_boxes.end());
  }

  std::sort(bbox_rects->begin(), bbox_rects->end(), cmp);

  // output
  int num_detected = keep_top_k < static_cast<int>(bbox_rects->size())
                         ? keep_top_k
                         : static_cast<int>(bbox_rects->size());
  bbox_rects->resize(num_detected);

  return num_detected;
}
}  // namespace mace



namespace mace {
namespace ops {

class DetectionOutput : public Operation {
 public:
  explicit DetectionOutput(OpConstructContext *context) : Operation(context) {}
};

template <RuntimeType D, class T>
class DetectionOutputOp;

template<class T>
class DetectionOutputOp<RuntimeType::RT_CPU, T> : public DetectionOutput {
 public:
  explicit DetectionOutputOp(OpConstructContext *context)
      : DetectionOutput(context),
        num_classes_(Operation::GetOptionalArg<int>("num_classes", 1)),
        nms_threshold_(Operation::GetOptionalArg<float>("nms_threshold", 0.3f)),
        nms_top_k_(Operation::GetOptionalArg<int>("nms_top_k", 100)),
        keep_top_k_(Operation::GetOptionalArg<int>("keep_top_k", 100)),
        confidence_threshold_(
            Operation::GetOptionalArg<float>("confidence_threshold", 0.05f)) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    Tensor *output = this->Output(0);

    auto *loc_t = this->Input(0);
    auto *conf_t = this->Input(1);
    auto *pbox_t = this->Input(2);


    auto num_prior = loc_t->shape()[1] / 4;

    MACE_CHECK(num_prior == conf_t->shape()[1] / num_classes_,
               "conf tensor shape miss match");
    MACE_CHECK(num_prior == pbox_t->shape()[2] / 4,
               "prior box tensor shape miss match");

    const float *loc_ptr = loc_t->data<float>();
    const float *conf_ptr = conf_t->data<float>();
    const float *pbox_ptr = pbox_t->data<float>();

    std::vector<BBox> bbox_rects;

    DetectionOutput_CLA(loc_ptr, conf_ptr, pbox_ptr, num_prior, num_classes_,
                        nms_threshold_, nms_top_k_, keep_top_k_,
                        confidence_threshold_, &bbox_rects);

    output->Clear();
    std::vector<index_t> output_shape = {1, 1, (int)bbox_rects.size(), 7};
    MACE_RETURN_IF_ERROR(output->Resize(output_shape))

    float *output_ptr = output->mutable_data<float>();
    auto output_len = (int)bbox_rects.size();
    for (int i = 0; i < output_len; ++i) {
      auto *row = output_ptr + i * 7;
      auto &b = bbox_rects[i];
      row[1] = b.label;
      row[2] = b.confidence;
      row[3] = b.xmin;
      row[4] = b.ymin;
      row[5] = b.xmax;
      row[6] = b.ymax;
    }

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  int num_classes_;
  float nms_threshold_;
  int nms_top_k_;
  int keep_top_k_;
  float confidence_threshold_;
};


void RegisterDetectionOutput(OpRegistry *op_registry) {
  MACE_REGISTER_OP(op_registry, "DetectionOutput", DetectionOutputOp,
                   RuntimeType::RT_CPU, float);

  MACE_REGISTER_OP_CONDITION(
      op_registry,
      OpConditionBuilder("DetectionOutput")
          .SetDevicePlacerFunc(
              [](OpConditionContext *context) -> std::set<RuntimeType> {
                return {RuntimeType::RT_CPU};
              }));
}

}  // namespace ops
}  // namespace mace
