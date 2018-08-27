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

package com.xiaomi.mace.demo;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;


import com.xiaomi.mace.demo.camera.CameraEngage;
import com.xiaomi.mace.demo.camera.CameraTextureView;
import com.xiaomi.mace.demo.camera.ContextMenuDialog;

import com.xiaomi.mace.demo.camera.MessageEvent;
import com.xiaomi.mace.demo.result.InitData;

import org.greenrobot.eventbus.EventBus;
import org.greenrobot.eventbus.Subscribe;
import org.greenrobot.eventbus.ThreadMode;

import java.util.Arrays;
import java.util.List;

public class CameraActivity extends Activity implements View.OnClickListener, AppModel.CreateEngineCallback {

    CameraEngage mCameraEngage;
    ImageView mPictureResult;
    Button mSelectMode;
    Button mSelectPhoneType;
    CameraTextureView mCameraTextureView;
    private TextView mResultView;
    private InitData initData = new InitData();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);
        setContentView(R.layout.activity_camera);
        mPictureResult = findViewById(R.id.iv_picture);
        mResultView = findViewById(R.id.tv_show_result);
        mCameraTextureView = findViewById(R.id.camera_texture);
        mCameraEngage = CameraFactory.genCameEngage(mCameraTextureView);

        mSelectMode = findViewById(R.id.tv_select_mode);
        mSelectMode.setOnClickListener(this);

        mSelectPhoneType = findViewById(R.id.tv_select_phone_type);
        mSelectPhoneType.setOnClickListener(this);
        initJni();
        initView();

    }

    @Override
    protected void onStart() {
        super.onStart();
        EventBus.getDefault().register(this);
    }

    @Override
    protected void onStop() {
        super.onStop();
        EventBus.getDefault().unregister(this);
    }

    private void initView() {
        mSelectMode.setText(initData.getModel());
        mSelectPhoneType.setText(initData.getDevice());
    }

    @Override
    protected void onResume() {
        super.onResume();
        mCameraEngage.onResume();
    }


    @Override
    protected void onPause() {
        super.onPause();
        mCameraEngage.onPause();
    }

    @Subscribe(threadMode = ThreadMode.MAIN)
    public void onPicture(MessageEvent.PicEvent picEvent) {
        if (picEvent != null && picEvent.getBitmap() != null) {
            mPictureResult.setImageBitmap(picEvent.getBitmap());
        }
    }

    @Subscribe(threadMode = ThreadMode.MAIN)
    public void onSetCameraViewSize(MessageEvent.OutputSizeEvent event) {
        if (mCameraTextureView != null) {
            mCameraTextureView.setRatio(event.width, event.height);
        }
    }


    @Subscribe(threadMode = ThreadMode.MAIN)
    public void onResultData(MessageEvent.MaceResultEvent resultData) {
        if (resultData != null && resultData.getData() != null) {
            String result = resultData.getData().name  + "\n"
                    + resultData.getData().probability + "\ncost time(ms): "
                    + resultData.getData().costTime;
            mResultView.setText(result);
        }
    }

    @Override
    public void onClick(View v) {
        switch (v.getId()) {
            case R.id.tv_select_mode:
                showSelectMode();
                break;
            case R.id.tv_select_phone_type:
                showPhoneType();
                break;
        }
    }

    private void initJni() {
        AppModel.instance.maceMobilenetSetAttrs(initData);
        AppModel.instance.maceMobilenetCreateEngine(initData, this);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == Constant.CAMERA_PERMISSION_REQ) {
            boolean allGrant = true;
            for (int result : grantResults) {
                if (result != PackageManager.PERMISSION_GRANTED) {
                    mCameraEngage.onResume();
                    allGrant = false;
                    break;
                }
            }
            if (allGrant) {
                initData = new InitData();
            }
        }
    }

    private void showPhoneType() {
        List<String> menus = Arrays.asList(InitData.DEVICES);
        ContextMenuDialog.show(this, menus, new ContextMenuDialog.OnClickItemListener() {
            @Override
            public void onCLickItem(String content) {
                mSelectPhoneType.setText(content);
                initData.setDevice(content);
                AppModel.instance.maceMobilenetCreateEngine(initData, CameraActivity.this);
            }
        });
    }

    private void showSelectMode() {
        List<String> menus = Arrays.asList(InitData.MODELS);
        ContextMenuDialog.show(this, menus, new ContextMenuDialog.OnClickItemListener() {
            @Override
            public void onCLickItem(String content) {
                mSelectMode.setText(content);
                initData.setModel(content);
                AppModel.instance.maceMobilenetCreateEngine(initData, CameraActivity.this);
            }
        });
    }

    @Override
    public void onCreateEngineFail(final boolean quit) {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Error");
        builder.setMessage("Failed to create inference engine with current setting:\n" + initData.getModel() + ", " + initData.getDevice());
        builder.setCancelable(false);
        builder.setPositiveButton(quit ? "Quit" : "Reset", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                if (quit) {
                    System.exit(0);
                } else {
                    dialog.dismiss();
                    resetCpu();
                }
            }
        });
        builder.show();
    }

    private void resetCpu() {
        String content = InitData.DEVICES[0];
        mSelectPhoneType.setText(content);
        initData.setDevice(content);
        AppModel.instance.maceMobilenetCreateEngine(initData, CameraActivity.this);
    }
}
