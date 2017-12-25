#!/bin/bash
set +x
Usage() {
  echo 'Usage: bash tools/create_mace_lib.sh tf_model_path image_size phone_version abi_version'
}

if [ $# -lt 4 ];then
  Usage
  exit -1
fi

IMAGE_SIZE=$2
PHONE_VERSION=$3
ABI_VERSION=$4
MACE_STATIC_LIB_DIR=libmace_${PHONE_VERSION}_gcn${IMAGE_SIZE}_${ABI_VERSION}
MACE_LIB_PATH=${MACE_STATIC_LIB_DIR}/lib/
MACE_INCLUDE_PATH=${MACE_STATIC_LIB_DIR}/include/mace/core/public/

rm -rf mace/codegen/models mace/codegen/opencl mace/codegen/opencl_bin mace/codegen/tuning mace/codegen/version
rm -rf ${MACE_STATIC_LIB_DIR}
mkdir -p ${MACE_LIB_PATH}
mkdir -p ${MACE_INCLUDE_PATH}

sh ./tools/validate_gcn.sh $1 $2
cp bazel-bin/mace/**/*.a ${MACE_LIB_PATH}
cp bazel-bin/mace/**/*.lo ${MACE_LIB_PATH}
cp mace/core/public/*.h ${MACE_INCLUDE_PATH}
