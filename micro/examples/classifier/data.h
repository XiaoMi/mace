// Copyright 2020 The MACE Authors. All Rights Reserved.
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

#ifndef MICRO_EXAMPLES_CLASSIFIER_DATA_H_
#define MICRO_EXAMPLES_CLASSIFIER_DATA_H_

#include "data/har.h"
#include "data/kws.h"
#include "data/mnist.h"
#include "stdint.h"

namespace mnist {
const float *input = data_mnist_4;
const int32_t input_dims[4] = {1, 28, 28, 1};
}  // namespace mnist

namespace har {
const float *input = data_har_standing;
const int32_t input_dims[4] = {1, 90, 3, 1};
}  // namespace har

namespace kws {
const float *input = data_kws_yes;
const int32_t input_dims[4] = {1, 98, 40, 1};
}  // namespace kws

#endif  // MICRO_EXAMPLES_CLASSIFIER_DATA_H_
