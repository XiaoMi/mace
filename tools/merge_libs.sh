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
MODEL_HEADER_DIR=${LIBMACE_BUILD_DIR}/libmace/include/mace/public
MODEL_DATA_DIR=${LIBMACE_BUILD_DIR}/libmace/data

rm -rf ${LIBMACE_BUILD_DIR}/libmace
mkdir -p ${LIBMACE_BUILD_DIR}/libmace/include/mace/public
mkdir -p ${LIBMACE_BUILD_DIR}/libmace/lib
mkdir -p ${MODEL_DATA_DIR}
cp -rf ${MACE_SOURCE_DIR}/mace/public/*.h ${LIBMACE_BUILD_DIR}/libmace/
cp ${MACE_SOURCE_DIR}/mace/core/runtime/hexagon/libhexagon_controller.so ${LIBMACE_BUILD_DIR}/libmace/lib

LIBMACE_TEMP_DIR=`mktemp -d -t libmace.XXXX`

# Merge all libraries in to one
echo "create ${LIBMACE_BUILD_DIR}/libmace/lib/libmace_${PROJECT_NAME}.a" > ${LIBMACE_TEMP_DIR}/libmace_${PROJECT_NAME}.mri

if [ x"$TARGET_ABI" = x"host" ]; then
  echo "addlib bazel-bin/mace/codegen/libgenerated_opencl_prod.pic.a" >> ${LIBMACE_TEMP_DIR}/libmace_${PROJECT_NAME}.mri
  echo "addlib bazel-bin/mace/codegen/libgenerated_tuning_params.pic.a" >> ${LIBMACE_TEMP_DIR}/libmace_${PROJECT_NAME}.mri
else
  echo "addlib bazel-bin/mace/codegen/libgenerated_opencl_prod.a" >> ${LIBMACE_TEMP_DIR}/libmace_${PROJECT_NAME}.mri
  echo "addlib bazel-bin/mace/codegen/libgenerated_tuning_params.a" >> ${LIBMACE_TEMP_DIR}/libmace_${PROJECT_NAME}.mri
  echo "addlib bazel-bin/mace/codegen/libgenerated_version.a" >> ${LIBMACE_TEMP_DIR}/libmace_${PROJECT_NAME}.mri
  echo "addlib bazel-bin/mace/core/libcore.a" >> ${LIBMACE_TEMP_DIR}/libmace_${PROJECT_NAME}.mri
  echo "addlib bazel-bin/mace/core/libopencl_prod.a" >> ${LIBMACE_TEMP_DIR}/libmace_${PROJECT_NAME}.mri
  echo "addlib bazel-bin/mace/kernels/libkernels.a" >> ${LIBMACE_TEMP_DIR}/libmace_${PROJECT_NAME}.mri
  echo "addlib bazel-bin/mace/utils/libutils.a" >> ${LIBMACE_TEMP_DIR}/libmace_${PROJECT_NAME}.mri
  echo "addlib bazel-bin/mace/utils/libutils_prod.a" >> ${LIBMACE_TEMP_DIR}/libmace_${PROJECT_NAME}.mri
fi

for model_output_dir in ${MODEL_OUTPUT_DIRS_ARR[@]}; do
  for lib in ${model_output_dir}/*.a; do
    echo "addlib ${lib}" >> ${LIBMACE_TEMP_DIR}/libmace_${PROJECT_NAME}.mri
  done
  for data_file in ${model_output_dir}/*.data; do
    cp ${data_file} ${MODEL_DATA_DIR}
  done
  for header_file in ${model_output_dir}/*.h; do
    cp ${header_file} ${MODEL_HEADER_DIR}
  done
done
echo "save" >> ${LIBMACE_TEMP_DIR}/libmace_${PROJECT_NAME}.mri
echo "end" >> ${LIBMACE_TEMP_DIR}/libmace_${PROJECT_NAME}.mri
$ANDROID_NDK_HOME/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64/bin/aarch64-linux-android-ar \
    -M < ${LIBMACE_TEMP_DIR}/libmace_${PROJECT_NAME}.mri || exit 1

rm -rf ${LIBMACE_TEMP_DIR}

echo "Libs merged!"
