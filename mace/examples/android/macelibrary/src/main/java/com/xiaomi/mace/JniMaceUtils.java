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

package com.xiaomi.mace;

public class JniMaceUtils {

    static {
        System.loadLibrary("mace_mobile_jni");
    }

    public static native int maceMobilenetSetAttrs(int ompNumThreads, int cpuAffinityPolicy, int gpuPerfHint, int gpuPriorityHint, String kernelPath);

    public static native int maceMobilenetCreateEngine(String model, String device);

    public static native float[] maceMobilenetClassify(float[] input);

}
