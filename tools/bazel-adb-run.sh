#!/bin/bash

if [ "$#" -lt 1 ]; then
  echo "$0" bazel-target [cmd params]
  exit 1
fi

MACE_SOURCE_DIR=`/bin/pwd`
CL_CODEGEN_DIR=${MACE_SOURCE_DIR}/mace/codegen/opencl
DEVICE_PATH=/data/local/tmp/mace
DEVICE_CL_PATH=$DEVICE_PATH/cl/
BAZEL_TARGET=$1
shift

# change //mace/a/b:c to bazel-bin/mace/a/b/c
BAZEL_BIN_PATH=`echo $BAZEL_TARGET | cut -d: -f1`
BAZEL_BIN_PATH=${BAZEL_BIN_PATH#//}
BAZEL_BIN_PATH=bazel-bin/$BAZEL_BIN_PATH
BIN_NAME=`echo $BAZEL_TARGET | cut -d: -f2`

ANDROID_ABI=arm64-v8a
ANDROID_ABI=armeabi-v7a
STRIP="--strip always"
VLOG_LEVEL=0
PROFILINE="--define profiling=true"

BRANCH=$(git symbolic-ref --short HEAD)
COMMIT_ID=$(git rev-parse --short HEAD)

echo "Step 1: Generate encrypted opencl source"
python mace/python/tools/encrypt_opencl_codegen.py \
    --cl_kernel_dir=./mace/kernels/opencl/cl/ --output_path=${CL_CODEGEN_DIR}/opencl_encrypt_program.cc

echo "Step 2: Build target"
bazel build -c opt $STRIP --verbose_failures $BAZEL_TARGET \
   --crosstool_top=//external:android/crosstool \
   --host_crosstool_top=@bazel_tools//tools/cpp:toolchain  \
   --cpu=$ANDROID_ABI \
   --define neon=false

if [ $? -ne 0 ]; then
  exit 1
fi

echo "Step 3: Run target"
du -hs $BAZEL_BIN_PATH/$BIN_NAME

for device in `adb devices | grep "^[A-Za-z0-9]\+[[:space:]]\+device$"| cut -f1`; do
  echo ======================================================================
  echo "Run on device: ${device}"
  adb -s ${device} shell "rm -rf $DEVICE_PATH"
  adb -s ${device} shell "mkdir -p $DEVICE_PATH"
  adb -s ${device} push $BAZEL_BIN_PATH/$BIN_NAME $DEVICE_PATH && \
  adb -s ${device} shell "MACE_KERNEL_PATH=$DEVICE_CL_PATH MACE_CPP_MIN_VLOG_LEVEL=$VLOG_LEVEL $DEVICE_PATH/$BIN_NAME $@"
done
