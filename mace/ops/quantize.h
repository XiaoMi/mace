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

#ifndef MACE_OPS_QUANTIZE_H_
#define MACE_OPS_QUANTIZE_H_

#include "mace/core/operator.h"
#include "mace/kernels/quantize.h"

namespace mace {
namespace ops {

template<DeviceType D, class T>
class QuantizeOp : public Operator<D, T> {
 public:
  QuantizeOp(const OperatorDef &operator_def, Workspace *ws)
    : Operator<D, T>(operator_def, ws) {
  }

  bool Run(StatsFuture *future) override {
    const Tensor *input = this->Input(INPUT);
    const Tensor *in_min = this->Input(IN_MIN);
    const Tensor *in_max = this->Input(IN_MAX);

    MACE_CHECK(in_min->size() == 1, "min val tensor has more than 1 value");
    MACE_CHECK(in_max->size() == 1, "max val tensor has more than 1 value");

    Tensor *output = this->Output(OUTPUT);
    Tensor *out_min = this->Output(OUT_MIN);
    Tensor *out_max = this->Output(OUT_MAX);
    output->ResizeLike(input);
    out_min->ResizeLike(in_min);
    out_max->ResizeLike(in_max);

    functor_(input, in_min, in_max, output, out_min, out_max, future);
    return true;
  }

 private:
  kernels::QuantizeFunctor<D, T> functor_;

 protected:
  OP_INPUT_TAGS(INPUT, IN_MIN, IN_MAX);
  OP_OUTPUT_TAGS(OUTPUT, OUT_MIN, OUT_MAX);
};

template<DeviceType D, class T>
class DequantizeOp : public Operator<D, T> {
 public:
  DequantizeOp(const OperatorDef &operator_def, Workspace *ws)
    : Operator<D, T>(operator_def, ws) {
  }

  bool Run(StatsFuture *future) override {
    const Tensor *input = this->Input(INPUT);
    const Tensor *in_min = this->Input(IN_MIN);
    const Tensor *in_max = this->Input(IN_MAX);

    MACE_CHECK(in_min->size() == 1, "min val tensor has more than 1 value");
    MACE_CHECK(in_max->size() == 1, "max val tensor has more than 1 value");

    Tensor *output = this->Output(OUTPUT);
    output->ResizeLike(input);

    functor_(input, in_min, in_max, output, future);
    return true;
  }

 private:
  kernels::DequantizeFunctor<D, T> functor_;

 protected:
  OP_INPUT_TAGS(INPUT, IN_MIN, IN_MAX);
  OP_OUTPUT_TAGS(OUTPUT);
};

template<DeviceType D, class T>
class RequantizeOp : public Operator<D, T> {
 public:
  RequantizeOp(const OperatorDef &operator_def, Workspace *ws)
    : Operator<D, T>(operator_def, ws) {
  }

  bool Run(StatsFuture *future) override {
    const Tensor *input = this->Input(INPUT);
    const Tensor *in_min = this->Input(IN_MIN);
    const Tensor *in_max = this->Input(IN_MAX);
    const Tensor *rerange_min = nullptr;
    const Tensor *rerange_max = nullptr;

    MACE_CHECK(in_min->size() == 1, "min val tensor has more than 1 value");
    MACE_CHECK(in_max->size() == 1, "max val tensor has more than 1 value");

    if (this->InputSize() >= 5) {
      rerange_min = this->Input(RERANGE_MIN);
      rerange_max = this->Input(RERANGE_MAX);
      MACE_CHECK(rerange_min->size() == 1,
                 "rerange min val tensor has more than 1 value");
      MACE_CHECK(rerange_max->size() == 1,
                 "rerange max val tensor has more than 1 value");
    }

    Tensor *output = this->Output(OUTPUT);
    Tensor *out_min = this->Output(OUT_MIN);
    Tensor *out_max = this->Output(OUT_MAX);
    output->ResizeLike(input);
    out_min->ResizeLike(in_min);
    out_max->ResizeLike(out_max);

    functor_(input,
             in_min,
             in_max,
             rerange_min,
             rerange_max,
             output,
             out_min,
             out_max,
             future);
    return true;
  }

 private:
  kernels::RequantizeFunctor<D, T> functor_;

 protected:
  OP_INPUT_TAGS(INPUT, IN_MIN, IN_MAX, RERANGE_MIN, RERANGE_MAX);
  OP_OUTPUT_TAGS(OUTPUT, OUT_MIN, OUT_MAX);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_QUANTIZE_H_
