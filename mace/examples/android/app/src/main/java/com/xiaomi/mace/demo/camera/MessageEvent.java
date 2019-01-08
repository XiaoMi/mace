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

package com.xiaomi.mace.demo.camera;

import android.graphics.Bitmap;

import com.xiaomi.mace.demo.result.ResultData;


public class MessageEvent {

    public static class MaceResultEvent {
        ResultData data;

        public MaceResultEvent(ResultData data) {
            this.data = data;
        }

        public ResultData getData() {
            return data;
        }
    }

    public static class PicEvent {
        Bitmap bitmap;

        public PicEvent(Bitmap bitmap) {
            this.bitmap = bitmap;
        }

        public Bitmap getBitmap() {
            return bitmap;
        }
    }

    public static class OutputSizeEvent {
        public int width;
        public int height;

        public OutputSizeEvent(int width, int height) {
            this.width = width;
            this.height = height;
        }
    }
}
