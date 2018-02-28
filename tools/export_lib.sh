#!/bin/bash

set -e

Usage() {
  echo "Usage: ./tools/export_lib.sh target_abi[armeabi-v7a | arm64-v8a | host] runtime[gpu | dsp] export_include_dir export_lib_dir"
  echo "eg: ./tools/export_lib.sh armeabi-v7a gpu ../include ../lib/libmace-armeabi-v7a"
}

if [ $# -lt 4 ]; then
  Usage
  exit 1
fi

TARGET_ABI=$1
RUNTIME=$2
EXPORT_INCLUDE_DIR=$3
EXPORT_LIB_DIR=$4

if [ x"${RUNTIME}" = x"dsp" ]; then
  DSP_MODE_BUILD_FLAGS="--define hexagon=true"
fi

MACE_SOURCE_DIR=`/bin/pwd`
CODEGEN_DIR=${MACE_SOURCE_DIR}/mace/codegen
CL_CODEGEN_DIR=${CODEGEN_DIR}/opencl
VERSION_CODEGEN_DIR=${CODEGEN_DIR}/version
STRIP="--strip always"

# TO REMOVE
LIBMACE_TEMP_DIR=`mktemp -d -t libmace.XXXX`

LIBMACE_NAME="libmace"
LIBMACE_DEV_NAME="libmace_dev"
LIBMACE_PROD_NAME="libmace_prod"

libmace_targets=(
  "//mace/ops:ops"
  "//mace/kernels:kernels"
  "//mace/codegen:generated_version"
  "//mace/core:core"
  "//mace/utils:utils"
)

libmace_dev_targets=(
  "//mace/codegen:generated_opencl_dev"
  "//mace/core:opencl_dev"
  "//mace/utils:utils_dev"
)

libmace_prod_targets=(
  "//mace/core:opencl_prod"
  "//mace/utils:utils_prod"
)

all_targets=(${libmace_targets[*]} ${libmace_dev_targets[*]} ${libmace_prod_targets[*]})

build_target()
{
  BAZEL_TARGET=$1
  bazel build --verbose_failures -c opt --strip always $BAZEL_TARGET \
    --crosstool_top=//external:android/crosstool \
    --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
     --cpu=$TARGET_ABI \
    --copt="-std=c++11" \
    --copt="-D_GLIBCXX_USE_C99_MATH_TR1" \
    --copt="-Werror=return-type" \
    --copt="-DMACE_OBFUSCATE_LITERALS" \
    --copt="-O3" \
    --define neon=true \
    --define openmp=true \
    $DSP_MODE_BUILD_FLAGS || exit 1
}

build_host_target()
{
  BAZEL_TARGET=$1
  bazel build --verbose_failures -c opt --strip always $BAZEL_TARGET \
    --copt="-std=c++11" \
    --copt="-D_GLIBCXX_USE_C99_MATH_TR1" \
    --copt="-Werror=return-type" \
    --copt="-DMACE_OBFUSCATE_LITERALS" \
    --copt="-O3" \
    --define openmp=true || exit 1
}

merge_libs()
{
  CREATE_LIB_NAME=$1
  LIBS_LIST=$2
  echo "create ${LIBMACE_TEMP_DIR}/${CREATE_LIB_NAME}.a" > ${LIBMACE_TEMP_DIR}/${CREATE_LIB_NAME}.mri || exit 1

  for lib_target in ${LIBS_LIST[*]}
  do
    lib_dir=`echo ${lib_target} | cut -d: -f1`
    lib_dir=${lib_dir#//}
    lib_name_prefix=lib`echo ${lib_target} | cut -d: -f2`
    bin_path="${MACE_SOURCE_DIR}/bazel-bin/${lib_dir}/${lib_name_prefix}"
    if [ x"${RUNTIME}" = x"local" ]; then
      if [ -f "${bin_path}.pic.a" ]; then
        bin_path="${bin_path}.pic.a"
      else
        bin_path="${bin_path}.pic.lo"
      fi
    else
      if [ -f "${bin_path}.a" ]; then
        bin_path="${bin_path}.a"
      else
        bin_path="${bin_path}.lo"
      fi
    fi
    echo "addlib ${bin_path}" >> ${LIBMACE_TEMP_DIR}/${CREATE_LIB_NAME}.mri || exit 1
  done

  echo "save" >> ${LIBMACE_TEMP_DIR}/${CREATE_LIB_NAME}.mri || exit 1
  echo "end" >> ${LIBMACE_TEMP_DIR}/${CREATE_LIB_NAME}.mri || exit 1

  $ANDROID_NDK_HOME/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64/bin/aarch64-linux-android-ar \
    -M < ${LIBMACE_TEMP_DIR}/${CREATE_LIB_NAME}.mri || exit 1
}


echo "Step 1: Generate encrypted opencl source"
python mace/python/tools/encrypt_opencl_codegen.py \
    --cl_kernel_dir=./mace/kernels/opencl/cl/ \
    --output_path=${CODEGEN_DIR}/opencl/opencl_encrypt_program.cc || exit 1


echo "Step 2: Generate version source"
rm -rf ${VERSION_CODEGEN_DIR}
mkdir ${VERSION_CODEGEN_DIR}
bash mace/tools/git/gen_version_source.sh ${CODEGEN_DIR}/version/version.cc || exit 1


echo "Step 3: Build libmace targets"
bazel clean
if [ x"${TARGET_ABI}" = x"host" ] || [ x"${TARGET_ABI}" = x"local" ]; then
  for target in ${all_targets[*]}
  do
    build_host_target ${target}
  done
else
  for target in ${all_targets[*]}
  do
    build_target ${target}
  done
fi


echo "Step 4: Create mri files and generate merged libs"
merge_libs "libmace" "${libmace_targets[*]}"
merge_libs "libmace_dev" "${libmace_dev_targets[*]}"
merge_libs "libmace_prod" "${libmace_prod_targets[*]}"


echo "Step 5: Export lib"
rm -rf ${EXPORT_INCLUDE_DIR}
mkdir -p ${EXPORT_INCLUDE_DIR}/mace/public
mkdir -p ${EXPORT_INCLUDE_DIR}/mace/utils
rm -rf ${EXPORT_LIB_DIR}
mkdir -p ${EXPORT_LIB_DIR}

cp ${MACE_SOURCE_DIR}/mace/public/*.h ${EXPORT_INCLUDE_DIR}/mace/public/ || exit 1
# utils is noti part of public API
cp ${MACE_SOURCE_DIR}/mace/utils/*.h ${EXPORT_INCLUDE_DIR}/mace/utils/ || exit 1
cp ${LIBMACE_TEMP_DIR}/libmace.a ${LIBMACE_TEMP_DIR}/libmace_dev.a ${LIBMACE_TEMP_DIR}/libmace_prod.a ${EXPORT_LIB_DIR}/ || exit 1

echo "Step 6: Remove temporary file"
rm -rf ${LIBMACE_TEMP_DIR}

echo "Done!"
