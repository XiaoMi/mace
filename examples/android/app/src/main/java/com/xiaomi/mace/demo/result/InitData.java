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

package com.xiaomi.mace.demo.result;

import android.os.Environment;

import java.io.File;

public class InitData {

    public static final String[] DEVICES = new String[]{"CPU", "GPU"};
    public static final String[] MODELS = new String[]{"mobilenet_v1", "mobilenet_v2", "mobilenet_v1_quant", "mobilenet_v2_quant"};
    private static final String[] ONLY_CPU_MODELS = new String[]{"mobilenet_v1_quant", "mobilenet_v2_quant"};

    private String model;
    private String device = "";
    private int ompNumThreads;
    private int cpuAffinityPolicy;
    private int gpuPerfHint;
    private int gpuPriorityHint;
    private String storagePath = "";

    public InitData() {
        model = MODELS[0];
        ompNumThreads = 2;
        cpuAffinityPolicy = 1;
        gpuPerfHint = 3;
        gpuPriorityHint = 3;
        device = DEVICES[0];
        storagePath = Environment.getExternalStorageDirectory().getAbsolutePath() + File.separator + "mace";
        File file = new File(storagePath);
        if (!file.exists()) {
            file.mkdir();
        }

    }

    public String getModel() {
        return model;
    }

    public void setModel(String model) {
        this.model = model;
    }

    public String getDevice() {
        return device;
    }

    public void setDevice(String device) {
        this.device = device;
    }

    public int getOmpNumThreads() {
        return ompNumThreads;
    }

    public void setOmpNumThreads(int ompNumThreads) {
        this.ompNumThreads = ompNumThreads;
    }

    public int getCpuAffinityPolicy() {
        return cpuAffinityPolicy;
    }

    public void setCpuAffinityPolicy(int cpuAffinityPolicy) {
        this.cpuAffinityPolicy = cpuAffinityPolicy;
    }

    public int getGpuPerfHint() {
        return gpuPerfHint;
    }

    public void setGpuPerfHint(int gpuPerfHint) {
        this.gpuPerfHint = gpuPerfHint;
    }

    public int getGpuPriorityHint() {
        return gpuPriorityHint;
    }

    public void setGpuPriorityHint(int gpuPriorityHint) {
        this.gpuPriorityHint = gpuPriorityHint;
    }

    public String getStoragePath() {
        return storagePath;
    }

    public void setStoragePath(String storagePath) {
        this.storagePath = storagePath;
    }

    public static String getCpuDevice() {
        return DEVICES[0];
    }

    public static boolean isOnlySupportCpuByModel(String model) {
        for (String m : ONLY_CPU_MODELS) {
            if (m.equals(model)) {
                return true;
            }
        }
        return false;
    }
}
