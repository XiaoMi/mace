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

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.SurfaceTexture;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.v4.content.ContextCompat;
import android.util.Log;
import android.view.TextureView.SurfaceTextureListener;
import android.view.View;

import com.xiaomi.mace.demo.AppModel;
import com.xiaomi.mace.demo.MaceApp;

import java.nio.FloatBuffer;

import static com.xiaomi.mace.demo.Constant.CAMERA_PERMISSION_REQ;

public abstract class CameraEngage implements SurfaceTextureListener {

    /**
     * preview height and width
     */
    int mPreviewHeight;
    int mPreviewWidth;

    /**
     * show camera view
     */
    CameraTextureView mTextureView;
    SurfaceTexture mSurfaceTexture;

    /**
     * switch camera use
     */
    private boolean mFacingFront = false;
    /**
     * camera background thread
     */
    private HandlerThread mBackgroundHandlerThread;
    private Handler mBackgroundHandler;

    /**
     * mace need data size width and height
     */
    private static final int FINAL_SIZE = 224;
    /**
     * storage rgb value
     */
    private int[] colorValues;

    /**
     * mace float[] input
     */
    private FloatBuffer floatBuffer;

    private final Object lock = new Object();

    private boolean isCapturePic = false;

    private Runnable mHandleCapturePicRunnable = new Runnable() {
        @Override
        public void run() {
            synchronized (lock) {
                if (isCapturePic) {
                    handleCapturePic();
                }
            }
            mBackgroundHandler.postDelayed(mHandleCapturePicRunnable, 200);
        }
    };

    private void handleCapturePic() {
        if (mTextureView != null) {
            Bitmap bitmap = mTextureView.getBitmap(FINAL_SIZE, FINAL_SIZE);
            if (bitmap != null) {
                bitmap.getPixels(colorValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
                handleColorRgbs();
//                EventBus.getDefault().post(new MessageEvent.PicEvent(bitmap));
                bitmap.recycle();
            }
        }
    }


    public CameraEngage(CameraTextureView mTextureView) {
        this.mTextureView = mTextureView;

        colorValues = new int[FINAL_SIZE * FINAL_SIZE];
        float[] floatValues = new float[FINAL_SIZE * FINAL_SIZE * 3];
        floatBuffer = FloatBuffer.wrap(floatValues, 0, FINAL_SIZE * FINAL_SIZE * 3);

        mPreviewHeight = MaceApp.app.getResources().getDisplayMetrics().heightPixels;
        mPreviewWidth = MaceApp.app.getResources().getDisplayMetrics().widthPixels;

        mTextureView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                autoFocus();
            }
        });
    }

    public void openCamera(int width, int height) {
        startCapturePic();
    }

    public abstract void autoFocus();

    public abstract void closeCamera();

    public abstract String getCameraId();

    public abstract void startPreview();

    boolean facingFrontPreview() {
        return mFacingFront;
    }

    public void setFacingFront(boolean facingFront) {
        this.mFacingFront = facingFront;
        onResume();
    }


    private void handleColorRgbs() {
        floatBuffer.rewind();
        for (int i = 0; i < colorValues.length; i++) {
            int value = colorValues[i];
            floatBuffer.put((((value >> 16) & 0xFF) - 128f)/ 128f);
            floatBuffer.put((((value >> 8) & 0xFF) - 128f) / 128f);
            floatBuffer.put(((value & 0xFF) - 128f) / 128f);
        }
        AppModel.instance.maceMobilenetClassify(floatBuffer.array());

    }


    public void onResume() {
        if (mTextureView.isAvailable()) {
            openCamera(mTextureView.getWidth(), mTextureView.getHeight());
        } else {
            mTextureView.setSurfaceTextureListener(this);
        }
    }

    private void startCapturePic() {
        mBackgroundHandlerThread = new HandlerThread("captureBackground");
        mBackgroundHandlerThread.start();
        mBackgroundHandler = new Handler(mBackgroundHandlerThread.getLooper());
        synchronized (lock) {
            isCapturePic = true;
        }

        mBackgroundHandler.post(mHandleCapturePicRunnable);
    }

    private void stopBackgroundThread() {
        try {
            mBackgroundHandlerThread.quitSafely();
            mBackgroundHandlerThread.join();
            mBackgroundHandler = null;
            mBackgroundHandlerThread = null;
            synchronized (lock) {
                isCapturePic = false;
            }
        } catch (Exception e) {
            Log.e(this.getClass().getName(), "stopBackgroundThread" + e);
        }
    }

    public void onPause() {
        closeCamera();
        stopBackgroundThread();
    }

    @Override
    public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
        mSurfaceTexture = surface;
        openCamera(width, height);
    }

    @Override
    public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {
    }

    @Override
    public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
        return false;
    }

    @Override
    public void onSurfaceTextureUpdated(SurfaceTexture surface) {

    }

    boolean checkCameraPermission() {
        if (ContextCompat.checkSelfPermission(mTextureView.getContext(), Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED || ContextCompat.checkSelfPermission(mTextureView.getContext(), Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED ||
                ContextCompat.checkSelfPermission(mTextureView.getContext(), Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ((Activity) mTextureView.getContext()).requestPermissions(new String[]{Manifest.permission.CAMERA, Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE}, CAMERA_PERMISSION_REQ);
            return false;
        }
        return true;
    }

}
