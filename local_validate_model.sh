#!/bin/bash
# Must run at root dir of mace project.
set +x
Usage() {
  echo 'Usage: bash tools/validate_model.sh tools/model.config'
}

if [ $# -lt 1 ];then
  Usage
  exit -1
fi

source $1

LIB_FOLDER_NAME="libmace_v7"

if [ x"$RUNTIME" = x"cpu" ]; then
  DATA_TYPE="DT_FLOAT"
  DEVICE_TYPE="CPU"
else
  Usage
  exit -1
fi

LIBMACE_TAG=`git describe --abbrev=0 --tags` || exit -1

VLOG_LEVEL=0
MODEL_DIR=$(dirname ${TF_MODEL_FILE_PATH})
LIBMACE_SOURCE_DIR=`/bin/pwd`
LIBMACE_BUILD_DIR="${LIBMACE_SOURCE_DIR}/build"
INPUT_FILE_NAME="model_input"
OUTPUT_FILE_NAME="model.out"
OUTPUT_LIST_FILE="model.list"
CODEGEN_DIR=${LIBMACE_SOURCE_DIR}/codegen
MODEL_CODEGEN_DIR=${CODEGEN_DIR}/models/${MODEL_TAG}
CL_CODEGEN_DIR=${CODEGEN_DIR}/opencl
CL_BIN_DIR=${CODEGEN_DIR}/opencl_bin
TUNING_CODEGEN_DIR=${CODEGEN_DIR}/tuning
VERSION_SOURCE_PATH=${CODEGEN_DIR}/version

build_and_run()
{
  PRODUCTION_MODE=$1
  if [ "$PRODUCTION_MODE" = true ]; then
    PRODUCTION_MODE_BUILD_FLAGS="--define production=true"
  fi

  round=1

  bazel build --verbose_failures -c opt --strip always examples:mace_run \
    --copt="-std=c++11" \
    --copt="-D_GLIBCXX_USE_C99_MATH_TR1" \
    --copt="-Werror=return-type" \
    --copt="-DMACE_MODEL_TAG=${MODEL_TAG}" \
    --define openmp=true \
    $PRODUCTION_MODE_BUILD_FLAGS || exit -1

  MACE_CPP_MIN_VLOG_LEVEL=$VLOG_LEVEL \
  bazel-bin/examples/mace_run \
      --input_shape="${INPUT_SHAPE}"\
      --output_shape="${OUTPUT_SHAPE}"\
      --input_file=${MODEL_DIR}/${INPUT_FILE_NAME} \
      --output_file=${MODEL_DIR}/${OUTPUT_FILE_NAME} \
      --device=${DEVICE_TYPE}   \
      --round=$round || exit -1
}

download_and_link_lib()
{
  if [ ! -d "${LIBMACE_SOURCE_DIR}/lib/${LIB_FOLDER_NAME}" ]; then
    wget -P ${LIBMACE_SOURCE_DIR}/lib http://cnbj1-inner-fds.api.xiaomi.net/libmace/libs/${LIBMACE_TAG}/${LIB_FOLDER_NAME}.tar.gz && \
      tar xvzf ${LIBMACE_SOURCE_DIR}/lib/${LIB_FOLDER_NAME}.tar.gz -C ${LIBMACE_SOURCE_DIR}/lib/ || exit -1
    echo "${LIB_FOLDER_NAME} download successfully!"
  else
    echo "${LIB_FOLDER_NAME} already exists!"
  fi

  echo "Create link 'mace' of downloaded or existed ${LIB_FOLDER_NAME}"
  if [ -L ${LIBMACE_SOURCE_DIR}/lib/mace ]; then
    unlink ${LIBMACE_SOURCE_DIR}/lib/mace
  fi
  ln -s ${LIBMACE_SOURCE_DIR}/lib/${LIB_FOLDER_NAME} ${LIBMACE_SOURCE_DIR}/lib/mace && \
    rm -rf ${LIBMACE_SOURCE_DIR}/lib/${LIB_FOLDER_NAME}.tar.gz || exit -1
}

echo "Step 1: Generate input data"
rm -rf ${MODEL_DIR}/${INPUT_FILE_NAME}
python tools/validate.py --generate_data true \
 --input_file=${MODEL_DIR}/${INPUT_FILE_NAME} \
 --input_shape="${INPUT_SHAPE}" || exit -1

echo "Step 2: Convert tf model to mace model and optimize memory"
bazel build //lib/python/tools:tf_converter || exit -1
rm -rf ${MODEL_CODEGEN_DIR}
mkdir -p ${MODEL_CODEGEN_DIR}
bazel-bin/lib/python/tools/tf_converter --input=${TF_MODEL_FILE_PATH} \
                                        --output=${MODEL_CODEGEN_DIR}/model.cc \
                                        --input_node=${TF_INPUT_NODE} \
                                        --output_node=${TF_OUTPUT_NODE} \
                                        --data_type=${DATA_TYPE} \
                                        --runtime=${RUNTIME} \
                                        --output_type=source \
                                        --template=${LIBMACE_SOURCE_DIR}/lib/python/tools/model.template \
                                        --model_tag=${MODEL_TAG} \
                                        --input_shape="${INPUT_SHAPE}" \
                                        --obfuscate=False || exit -1

echo "Step 3: Download mace static library"
download_and_link_lib

echo "Step 4: remove the mace run result."
rm -rf ${MODEL_DIR}/${OUTPUT_FILE_NAME}

echo "Step 7: Run model on the phone using binary"
build_and_run true

echo "Step 9: Validate the result"
python tools/validate.py --model_file ${TF_MODEL_FILE_PATH} \
    --input_file ${MODEL_DIR}/${INPUT_FILE_NAME} \
    --mace_out_file ${MODEL_DIR}/${OUTPUT_FILE_NAME} \
    --mace_runtime ${RUNTIME} \
    --input_node ${TF_INPUT_NODE} \
    --output_node ${TF_OUTPUT_NODE} \
    --input_shape ${INPUT_SHAPE} \
    --output_shape ${OUTPUT_SHAPE}

echo "Step 10: Generate project static lib"
rm -rf ${LIBMACE_BUILD_DIR}
mkdir -p ${LIBMACE_BUILD_DIR}/lib
cp -rf ${LIBMACE_SOURCE_DIR}/include ${LIBMACE_BUILD_DIR}

echo "Done"
