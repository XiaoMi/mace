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

package com.xiaomi.mace.demo;

import android.os.Build;

import com.xiaomi.mace.demo.camera.CameraApiLessM;
import com.xiaomi.mace.demo.camera.CameraEngage;
import com.xiaomi.mace.demo.camera.CameraTextureView;

public class CameraFactory {

    public static CameraEngage genCameEngage(CameraTextureView textureView) {
        CameraEngage cameraEngage;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
//            cameraEngage = new CameraApiMoreM(textureView);
            cameraEngage = new CameraApiLessM(textureView);
        } else {
            cameraEngage = new CameraApiLessM(textureView);
        }
        return cameraEngage;
    }
}
