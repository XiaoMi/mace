#!/bin/bash

if [ "$#" -lt 1 ]; then
  echo "$0" bazel-target [cmd params]
  exit 1
fi

MACE_SOURCE_DIR=`/bin/pwd`
CODEGEN_DIR=${MACE_SOURCE_DIR}/mace/codegen
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
PROFILING="1"

echo "Step 1: Generate encrypted opencl source"
python mace/python/tools/encrypt_opencl_codegen.py \
    --cl_kernel_dir=./mace/kernels/opencl/cl/ --output_path=${CODEGEN_DIR}/opencl/opencl_encrypt_program.cc

echo "Step 2: Generate version source"
mkdir -p ${CODEGEN_DIR}/version
bash mace/tools/git/gen_version_source.sh ${CODEGEN_DIR}/version/version.cc

echo "Step 3: Build target"
# -D_GLIBCXX_USE_C99_MATH_TR1 is used to solve include error instead
# of linking error which solved by -lm
bazel build -c opt $STRIP --verbose_failures $BAZEL_TARGET \
   --crosstool_top=//external:android/crosstool \
   --host_crosstool_top=@bazel_tools//tools/cpp:toolchain  \
   --cpu=$ANDROID_ABI \
   --copt="-std=c++11" \
   --copt="-D_GLIBCXX_USE_C99_MATH_TR1" \
   --copt="-DMACE_DISABLE_NO_TUNING_WARNING" \
   --copt="-Werror=return-type" \
   --define neon=false

if [ $? -ne 0 ]; then
  exit 1
fi

echo "Step 4: Run target"
du -hs $BAZEL_BIN_PATH/$BIN_NAME

for device in `adb devices | grep "^[A-Za-z0-9]\+[[:space:]]\+device$"| cut -f1`; do
  echo ======================================================================
  echo "Run on device: ${device}"
  adb -s ${device} shell "rm -rf $DEVICE_PATH"
  adb -s ${device} shell "mkdir -p $DEVICE_PATH"
  adb -s ${device} shell "mkdir -p $DEVICE_PATH/cl"
  adb -s ${device} push $BAZEL_BIN_PATH/$BIN_NAME $DEVICE_PATH && \
  adb -s ${device} shell "MACE_OPENCL_PROFILING=$PROFILING MACE_KERNEL_PATH=$DEVICE_CL_PATH MACE_CPP_MIN_VLOG_LEVEL=$VLOG_LEVEL $DEVICE_PATH/$BIN_NAME $@"
done
