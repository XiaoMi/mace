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

import android.graphics.ImageFormat;
import android.hardware.Camera;
import android.text.TextUtils;
import android.util.Log;

import java.util.List;

public class CameraApiLessM extends CameraEngage implements Camera.AutoFocusCallback {

    private Camera mCamera;

    public CameraApiLessM(CameraTextureView textureView) {
        super(textureView);
    }

    @Override
    public void openCamera(int width, int height) {
        if (!checkCameraPermission()) {
            return;
        }
        super.openCamera(width, height);
        closeCamera();
        String cameraId = getCameraId();
        if (TextUtils.isEmpty(cameraId)) {
            return;
        }

        mCamera = Camera.open(Integer.parseInt(cameraId));
        setOutputConfig(width, height);
        startPreview();
    }

    @Override
    public void autoFocus() {
        doAutoFocus();
    }

    @Override
    public void closeCamera() {
        if (mCamera != null) {
            mCamera.stopPreview();
            mCamera.setPreviewCallback(null);
            mCamera.release();
            mCamera = null;
        }
    }

    @Override
    public String getCameraId() {
        Camera.CameraInfo cameraInfo = new Camera.CameraInfo();
        int cameraFacing = facingFrontPreview() ? Camera.CameraInfo.CAMERA_FACING_FRONT : Camera.CameraInfo.CAMERA_FACING_BACK;
        for (int i = 0; i < Camera.getNumberOfCameras(); i++) {
            Camera.getCameraInfo(i, cameraInfo);
            if (cameraInfo.facing == cameraFacing) {
                return String.valueOf(i);
            }
        }
        return "";
    }

    @Override
    public void startPreview() {
        try {
            doAutoFocus();
            mCamera.setPreviewTexture(mSurfaceTexture);
            mCamera.startPreview();
            mCamera.setDisplayOrientation(90);
        } catch (Exception e) {
            Log.e(getClass().getName(), "startPreview error = " + e);
        }
    }

    @Override
    public void onAutoFocus(boolean success, Camera camera) {
        if (success) {
            mCamera.cancelAutoFocus();
        }
    }

    private void doAutoFocus() {
        try {
            mCamera.autoFocus(this);
        } catch (Throwable e) {
            Log.e(this.getClass().getName(), "auto focus error = " + e);
        }
    }

    private void setOutputConfig(int width, int height) {
        Camera.Parameters parameters = mCamera.getParameters();
        parameters.setFocusMode(Camera.Parameters.FOCUS_MODE_AUTO);
        Camera.Size size = getOptimalSize(parameters.getSupportedPreviewSizes(), width, height);
        mPreviewWidth = size.height;
        mPreviewHeight = size.width;
        parameters.setPreviewSize(size.width, size.height);
        mCamera.setParameters(parameters);
        mTextureView.setRatio(mPreviewWidth, mPreviewHeight);
    }

    private Camera.Size getOptimalSize(List<Camera.Size> sizes, int w, int h) {
        final double ASPECT_TOLERANCE = 0.1;
        double targetRatio = (double) w / h;
        Camera.Size optimalSize = null;
        double minDiff = Double.MAX_VALUE;

        int targetHeight = h;

        for (Camera.Size size : sizes) {
            double ratio = (double) size.height / size.width;
            if (Math.abs(ratio - targetRatio) > ASPECT_TOLERANCE) {
                continue;
            }
            if (Math.abs(size.height - targetHeight) < minDiff) {
                optimalSize = size;
                minDiff = Math.abs(size.height - targetHeight);
            }
        }

        if (optimalSize == null) {
            minDiff = Double.MAX_VALUE;
            for (Camera.Size size : sizes) {
                if (Math.abs(size.height - targetHeight) < minDiff) {
                    optimalSize = size;
                    minDiff = Math.abs(size.height - targetHeight);
                }
            }
        }

        return optimalSize;
    }
}
