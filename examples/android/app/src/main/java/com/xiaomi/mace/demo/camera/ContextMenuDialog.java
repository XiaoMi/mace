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

import android.app.Activity;
import android.app.DialogFragment;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ListView;

import com.xiaomi.mace.demo.R;

import java.util.ArrayList;
import java.util.List;


public class ContextMenuDialog extends DialogFragment {

    private List<String> items = new ArrayList<>();
    private OnClickItemListener listener;

    public static void show(Activity activity, List<String> items, OnClickItemListener listener) {
        ContextMenuDialog dialog = new ContextMenuDialog();
        dialog.items = items;
        dialog.listener = listener;
        dialog.show(activity.getFragmentManager(), "");
    }

    @Nullable
    @Override
    public View onCreateView(LayoutInflater inflater, @Nullable ViewGroup container, Bundle savedInstanceState) {
        super.onCreateView(inflater, container, savedInstanceState);
        View root = inflater.inflate(R.layout.layout_dialog, null);
        final ListView listView = root.findViewById(R.id.list_menu);
        ItemAdapter itemAdapter = new ItemAdapter();
        listView.setAdapter(itemAdapter);
        itemAdapter.setItems(items);
        listView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                if (listener != null) {
                    listener.onCLickItem(items.get(position));
                }
                dismissAllowingStateLoss();
            }
        });
        return root;
    }

    public static interface OnClickItemListener {
        void onCLickItem(String content);
    }
}
