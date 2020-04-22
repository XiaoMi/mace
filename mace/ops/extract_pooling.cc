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

// This Op is for fused StatisticsExtraction and StatisticsPooling
// Components in Kaldi.
// This op is used to extract moving-average mean and standard-deviation
// statistics of input data.
// 'forward_indexes' indicates which frames of input will be used for
// extraction.
// save statistics results.
// 'forward_indexes' and 'count' were from precomputed index in kaldi.
// Reference to tools/extract_pooling.py and
// http://kaldi-asr.org/doc/nnet-general-component_8h_source.html#l00158

#include <functional>
#include <memory>

#include "mace/core/ops/operator.h"
#include "mace/core/registry/ops_registry.h"


namespace mace {
namespace ops {

template <DeviceType D, typename T>
class ExtractPoolingOp;

template <typename T>
class ExtractPoolingOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit ExtractPoolingOp(OpConstructContext *context)
      : Operation(context),
        include_variance_(
            static_cast<bool>(
                Operation::GetOptionalArg<int>("include_variance", 0))),
        num_log_count_(
            Operation::GetOptionalArg<int>("num_log_count", 0)),
        variance_floor_(
            Operation::GetOptionalArg<float>("variance_floor", 1.0e-10)),
        forward_indexes_(Operation::GetRepeatedArgs<int>("forward_indexes")),
        counts_(Operation::GetRepeatedArgs<float>("counts")) {}

  inline void Validate() {
    const Tensor *input = this->Input(0);
    const unsigned int rank = static_cast<unsigned int>(input->dim_size());
    MACE_CHECK(rank >= 2,
               "ExtractPooling only supports input dim size >= 2");
    MACE_CHECK(counts_.size() * 2 == forward_indexes_.size(),
               "counts length(", counts_.size(),
               ") should be 2 times of forward_indexes length(",
               forward_indexes_.size(), ").");
    for (size_t i = 0; i < counts_.size(); ++i) {
      MACE_CHECK(static_cast<index_t>(counts_[i]) ==
                     forward_indexes_[2 * i + 1] - forward_indexes_[2 * i],
                 "invalid forward indexes and counts values");
    }
  }

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);
    Validate();
    const std::vector<index_t> &input_shape = input->shape();
    const unsigned int dim_size = static_cast<unsigned int>(input->dim_size());

    const index_t input_dim = input_shape[dim_size - 1];
    const index_t chunk = input_shape[dim_size - 2];
    const index_t output_chunk = counts_.size();
    const index_t batch =
        std::accumulate(input_shape.begin(), input_shape.end() - 2, 1,
                        std::multiplies<index_t>());

    index_t output_dim = include_variance_ ? 2 * input_dim : input_dim;
    output_dim += num_log_count_;
    std::vector<index_t> output_shape(input_shape);
    output_shape[dim_size - 1] = output_dim;
    output_shape[dim_size - 2] = output_chunk;
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));

    const index_t extract_out_size = PadAlignSize(output_dim * sizeof(T));
    ScratchBuffer *scratch = context->device()->scratch_buffer();
    scratch->Rewind();
    scratch->GrowSize(extract_out_size);

    Tensor extract_out(
        scratch->Scratch(extract_out_size), DataTypeToEnum<T>::v());
    extract_out.Reshape({1, output_dim});
    extract_out.Clear();
    T *extract_out_data = extract_out.mutable_data<T>();

    Tensor::MappingGuard guard_input(input);
    Tensor::MappingGuard guard_output(output);
    const T *input_data = input->data<T>();
    T *output_data = output->mutable_data<T>();

    utils::ThreadPool
        &thread_pool = context->device()->cpu_runtime()->thread_pool();

    for (index_t b = 0; b < batch; ++b) {
      for (index_t i = 0; i < output_chunk; ++i) {
        int start = forward_indexes_[2 * i];
        int end = forward_indexes_[2 * i + 1];
        float count = counts_[i];
        float mean_scale = 1.f / count;
        float log_count = std::log(count);
        thread_pool.Compute1D([=](index_t start0,
                                  index_t end0,
                                  index_t step0) {
          for (index_t n = start0; n < end0; n += step0) {
            extract_out_data[n] = log_count;
          }
        }, 0, num_log_count_, 1);
        if (include_variance_) {
          thread_pool.Compute1D([=](index_t start0,
                                    index_t end0,
                                    index_t step0) {
            for (index_t d = start0; d < end0; d += step0) {
              float mean = 0.f;
              float variance = 0.f;
              for (int t = start; t < end; ++t) {
                index_t input_index =
                    (b * chunk + t)
                        * input_dim;
                float x = input_data[input_index + d];
                mean += x;
                variance += x * x;
              }
              mean *= mean_scale;
              variance *= mean_scale;
              extract_out_data[d + num_log_count_] = mean;
              variance = variance - mean * mean;
              extract_out_data[d + input_dim + num_log_count_] =
                  variance < variance_floor_ ?
                  std::sqrt(variance_floor_) :
                  std::sqrt(variance);
            }
          }, 0, input_dim, 1);
        } else {
          thread_pool.Compute1D([=](index_t start0,
                                    index_t end0,
                                    index_t step0) {
            for (index_t d = start0; d < end0; d += step0) {
              float mean = 0.f;
              for (int t = start; t < end; ++t) {
                index_t input_index =
                    (b * chunk + t) * input_dim;
                mean += input_data[input_index + d];
              }
              extract_out_data[d + num_log_count_] = mean * mean_scale;
            }
          }, 0, input_dim, 1);
        }
        memcpy(output_data + (b * output_chunk + i) * output_dim,
               extract_out_data, output_dim * sizeof(T));
      }
    }

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  bool include_variance_;
  int num_log_count_;
  float variance_floor_;
  std::vector<int> forward_indexes_;
  std::vector<float> counts_;
};

void RegisterExtractPooling(OpRegistry *op_registry) {
  MACE_REGISTER_OP(op_registry, "ExtractPooling", ExtractPoolingOp,
                   DeviceType::CPU, float);
  MACE_REGISTER_BF16_OP(op_registry, "ExtractPooling", ExtractPoolingOp,
                        DeviceType::CPU);
}

}  // namespace ops
}  // namespace mace
