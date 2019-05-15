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

import android.content.res.AssetManager;
import android.util.Log;

import com.xiaomi.mace.demo.MaceApp;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class LabelCache {

    private static LabelCache labelCache;
    private LabelCache() {
        readCacheLabelFromLocalFile();
    }
    private List<Float> floatList = new ArrayList<>();
    private List<String> resultLabel = new ArrayList<>();
    private ResultData mResultData;

    public static LabelCache instance() {
        if (labelCache == null) {
            synchronized (LabelCache.class) {
                if (labelCache == null) {
                    labelCache = new LabelCache();
                }
            }
        }
        return labelCache;
    }


    private void readCacheLabelFromLocalFile() {
        try {
            AssetManager assetManager = MaceApp.app.getApplicationContext().getAssets();
            BufferedReader reader = new BufferedReader(new InputStreamReader(assetManager.open("cacheLabel.txt")));
            String readLine = null;
            while ((readLine = reader.readLine()) != null) {
                Log.d("labelCache", "readLine = " + readLine);
                resultLabel.add(readLine);
            }
            reader.close();
        } catch (Exception e) {
            Log.e("labelCache", "error " + e);
        }
    }

    public ResultData getResultFirst(float[] floats) {
        floatList.clear();
        for (float f : floats) {
            floatList.add(f);
        }
        float maxResult = Collections.max(floatList);

        int indexResult = floatList.indexOf(maxResult);
        if (indexResult < resultLabel.size()) {
            String result = resultLabel.get(indexResult);
            if (result != null) {
                if (mResultData == null) {
                    mResultData = new ResultData(result, maxResult);
                } else {
                    mResultData.updateData(result, maxResult);
                }
                return mResultData;
            }
        }
        return null;


    }


}
