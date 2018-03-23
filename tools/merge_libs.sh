#!/bin/bash

Usage() {
  echo "Usage: bash tools/merge_libs.sh target_soc libmace_output_dir model_output_dirs"
}

if [ $# -lt 3 ]; then
  Usage
  exit 1
fi

CURRENT_DIR=`dirname $0`
source ${CURRENT_DIR}/env.sh

TARGET_SOC=$1
LIBMACE_BUILD_DIR=$2
MODEL_OUTPUT_DIRS=$3
MODEL_OUTPUT_DIRS_ARR=(${MODEL_OUTPUT_DIRS//,/ })
MODEL_HEADER_DIR=${LIBMACE_BUILD_DIR}/include/mace/public
MODEL_DATA_DIR=${LIBMACE_BUILD_DIR}/data

if [ ! -d "${MODEL_HEADER_DIR}" ]; then
  mkdir -p ${MODEL_HEADER_DIR}
fi
cp -rf ${MACE_SOURCE_DIR}/mace/public/*.h ${LIBMACE_BUILD_DIR}/include/mace/public/

if [ ! -d "${LIBMACE_BUILD_DIR}/${TARGET_ABI}" ]; then
  mkdir -p ${LIBMACE_BUILD_DIR}/${TARGET_ABI}
fi
if [ ! -d "${MODEL_DATA_DIR}" ]; then
  mkdir -p ${MODEL_DATA_DIR}
fi

if [ x"${TARGET_ABI}" = x"armeabi-v7a" ]; then
  cp ${MACE_SOURCE_DIR}/mace/core/runtime/hexagon/libhexagon_controller.so ${LIBMACE_BUILD_DIR}/${TARGET_ABI}/
fi

LIBMACE_TEMP_DIR=`mktemp -d -t libmace.XXXX`

# Merge all libraries in to one
echo "create ${LIBMACE_BUILD_DIR}/${TARGET_ABI}/libmace_${PROJECT_NAME}.${TARGET_SOC}.a" > ${LIBMACE_TEMP_DIR}/libmace_${PROJECT_NAME}.mri

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
  echo "addlib bazel-bin/mace/ops/libops.lo" >> ${LIBMACE_TEMP_DIR}/libmace_${PROJECT_NAME}.mri
fi

for model_output_dir in ${MODEL_OUTPUT_DIRS_ARR[@]}; do
  for lib in ${model_output_dir}/*.a; do
    echo "addlib ${lib}" >> ${LIBMACE_TEMP_DIR}/libmace_${PROJECT_NAME}.mri
  done
  if [ "${EMBED_MODEL_DATA}" == "0" ];then
    for data_file in ${model_output_dir}/*.data; do
      cp ${data_file} ${MODEL_DATA_DIR}
    done
  fi
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
