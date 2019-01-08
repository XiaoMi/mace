# Copyright 2018 The MACE Authors. All Rights Reserved.
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

import onnx
import sys
from onnx import optimizer


# Usage: python onnx_optimizer.py model.onnx model_opt.onnx


def main():
    if len(sys.argv) != 3:
        print "Usage: python onnx_optimizer.py model.onnx model_opt.onnx"
        sys.exit(0)
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    original_model = onnx.load(in_path)
    print "Start optimize ONNX model for inference:"
    passes = ['eliminate_identity',
              'fuse_consecutive_squeezes',
              'fuse_consecutive_transposes',
              'eliminate_nop_pad',
              'eliminate_nop_transpose',
              'eliminate_unused_initializer',
              'extract_constant_to_initializer',
              'fuse_add_bias_into_conv',
              'fuse_bn_into_conv',
              'fuse_transpose_into_gemm']
    for i in range(len(passes)):
        print i, ".", passes[i]
    optimized_model = optimizer.optimize(original_model, passes)
    onnx.save_model(optimized_model, out_path)
    print "Optimize Finished!"
    print "Please check new model in:", out_path


if __name__ == '__main__':
    main()
