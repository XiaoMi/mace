#!/bin/bash

Usage() {
  echo "Usage: bash tools/merge_libs.sh libmace_output_dir model_output_dirs"
}

if [ $# -lt 2 ]; then
  Usage
  exit 1
fi

CURRENT_DIR=`dirname $0`
source ${CURRENT_DIR}/env.sh

LIBMACE_BUILD_DIR=$1
MODEL_OUTPUT_DIRS=$2
MODEL_OUTPUT_DIRS_ARR=(${MODEL_OUTPUT_DIRS//,/ })

rm -rf ${LIBMACE_BUILD_DIR}/libmace
mkdir -p ${LIBMACE_BUILD_DIR}/libmace/lib
cp -rf ${LIBMACE_SOURCE_DIR}/include ${LIBMACE_BUILD_DIR}/libmace/
for model_output_dir in ${MODEL_OUTPUT_DIRS_ARR[@]}
do
  cp ${model_output_dir}/*.a ${LIBMACE_BUILD_DIR}/libmace/lib/
done
cp ${LIBMACE_SOURCE_DIR}/lib/hexagon/libhexagon_controller.so ${LIBMACE_BUILD_DIR}/libmace/lib

LIBMACE_TEMP_DIR=`mktemp -d -t libmace.XXXX`

# Merge libmace engine
echo "create ${LIBMACE_BUILD_DIR}/libmace/lib/libmace_engine.a" > ${LIBMACE_TEMP_DIR}/libmace_engine.mri
echo "addlib lib/mace/libmace.a" >> ${LIBMACE_TEMP_DIR}/libmace_engine.mri
echo "addlib lib/mace/libmace_prod.a" >> ${LIBMACE_TEMP_DIR}/libmace_engine.mri
echo "save" >> ${LIBMACE_TEMP_DIR}/libmace_engine.mri
echo "end" >> ${LIBMACE_TEMP_DIR}/libmace_engine.mri
$ANDROID_NDK_HOME/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64/bin/aarch64-linux-android-ar \
    -M < ${LIBMACE_TEMP_DIR}/libmace_engine.mri || exit 1

# Merge opencl bin and tuning result
echo "create ${LIBMACE_BUILD_DIR}/libmace/lib/libmace_opencl.a" > ${LIBMACE_TEMP_DIR}/libmace_opencl.mri
if [ x"TARGET_ABI" = x"host" ]; then
  echo "addlib bazel-bin/codegen/libgenerated_opencl_prod.pic.a" >> ${LIBMACE_TEMP_DIR}/libmace_opencl.mri
  echo "addlib bazel-bin/codegen/libgenerated_tuning_params.pic.a" >> ${LIBMACE_TEMP_DIR}/libmace_opencl.mri
else
  echo "addlib bazel-bin/codegen/libgenerated_opencl_prod.a" >> ${LIBMACE_TEMP_DIR}/libmace_opencl.mri
  echo "addlib bazel-bin/codegen/libgenerated_tuning_params.a" >> ${LIBMACE_TEMP_DIR}/libmace_opencl.mri
fi
echo "save" >> ${LIBMACE_TEMP_DIR}/libmace_opencl.mri
echo "end" >> ${LIBMACE_TEMP_DIR}/libmace_opencl.mri
$ANDROID_NDK_HOME/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64/bin/aarch64-linux-android-ar \
    -M < ${LIBMACE_TEMP_DIR}/libmace_opencl.mri || exit 1

rm -rf ${LIBMACE_TEMP_DIR}

echo "Libs merged!"
