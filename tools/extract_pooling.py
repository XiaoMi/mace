# Copyright 2019 The MACE Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import math

variance_floor = 1.0e-10

input_data = np.arange(180).reshape(3, 20, 3).astype(np.float32)
print("input data:", input_data)
num_log_count = 0
include_var = 0
forward_indexes = [0, 6, 2, 6]
counts = [6, 4]

input_dim = input_data.shape[-1]
input_chunk = input_data.shape[-2]

out_chunk = len(counts)
batch = input_data.size / (input_dim * input_chunk)
input_data.reshape(batch, input_chunk, input_dim)

output_dim = input_dim

if include_var > 0:
    output_dim += input_dim
if num_log_count > 0:
    output_dim += num_log_count

output_data = np.zeros((batch, out_chunk, output_dim), dtype=np.float32)

for b in range(0, batch):
    for i in range(0, out_chunk):
        start = forward_indexes[2 * i]
        end = forward_indexes[2 * i + 1]
        count = counts[i]
        mean_scale = 1.0 / count
        log_count = math.log(count)
        if num_log_count > 0:
            for n in range(0, num_log_count):
                output_data[b, i, n] = log_count
        for d in range(0, input_dim):
            mean = 0.0
            variance = 0.0
            for t in range(start, end):
                x = input_data[b, t, d]
                mean += x
                variance += x * x
            mean = mean * mean_scale
            output_data[b, i, d + num_log_count] = mean
            if include_var > 0:
                variance = variance * mean_scale - mean * mean
                idx = d + input_dim + num_log_count
                if variance < variance_floor:
                    output_data[b, i, idx] = math.sqrt(variance_floor)
                else:
                    output_data[b, i, idx] = math.sqrt(variance)
print("output data:", output_data)
print("output data shape:", output_data.shape)
