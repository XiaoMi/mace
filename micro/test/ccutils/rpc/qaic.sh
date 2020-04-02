#!/usr/bin/env bash

output_dir=${1}
mkdir -p output_dir
echo $HEXAGON_SDK_ROOT/tools/qaic/Ubuntu16/qaic \
     		-mdll -o ${output_dir} \
     		-I$HEXAGON_SDK_ROOT/libs/fastcv/dspCV/android_Debug/ship \
     		-I$HEXAGON_SDK_ROOT/libs/common/rpcmem/android_Debug/ship \
     		-I$HEXAGON_SDK_ROOT/libs/common/adspmsgd/ship/android_Debug \
     		-I$HEXAGON_SDK_ROOT/incs \
     		-I$HEXAGON_SDK_ROOT/libs/common/remote/ship/android_Debug \
     		-I$HEXAGON_SDK_ROOT/incs/stddef \
     		${@:2}
$HEXAGON_SDK_ROOT/tools/qaic/Ubuntu16/qaic \
		-mdll -o ${output_dir} \
		-I$HEXAGON_SDK_ROOT/libs/fastcv/dspCV/android_Debug/ship \
		-I$HEXAGON_SDK_ROOT/libs/common/rpcmem/android_Debug/ship \
		-I$HEXAGON_SDK_ROOT/libs/common/adspmsgd/ship/android_Debug \
		-I$HEXAGON_SDK_ROOT/incs \
		-I$HEXAGON_SDK_ROOT/libs/common/remote/ship/android_Debug \
		-I$HEXAGON_SDK_ROOT/incs/stddef \
		${@:2}
