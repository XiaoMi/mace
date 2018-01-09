#!/bin/bash
# Must run at root dir of mace project.
set +x
Usage() {
  echo 'Usage: bash tools/validate_gcn.sh tools/gcn.config tf_model_path model_tag image_size [tuning]'
}

if [ $# -lt 4 ];then
  Usage
  exit -1
fi

source $1

TF_MODEL_FILE_PATH=$2
MODEL_TAG=$3
IMAGE_SIZE=$4
TUNING_OR_NOT=${5:-0}

VLOG_LEVEL=0
MODEL_DIR=$(dirname ${TF_MODEL_FILE_PATH})
MACE_SOURCE_DIR=`/bin/pwd`
INPUT_FILE_NAME='model_input'
OUTPUT_FILE_NAME='gcn.out'
OUTPUT_LIST_FILE='gcn.list'
PHONE_DATA_DIR="/data/local/tmp/${MODEL_TAG}"
KERNEL_DIR="${PHONE_DATA_DIR}/cl/"
CODEGEN_DIR=${MACE_SOURCE_DIR}/mace/codegen
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

  bazel build --verbose_failures -c opt --strip always mace/examples:mace_run \
    --crosstool_top=//external:android/crosstool \
    --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
    --cpu=arm64-v8a \
    --copt="-std=c++11" \
    --copt="-D_GLIBCXX_USE_C99_MATH_TR1" \
    --copt="-Werror=return-type" \
    --copt="-DMACE_MODEL_TAG=${MODEL_TAG}" \
    $PRODUCTION_MODE_BUILD_FLAGS  || exit -1

  adb shell "mkdir -p ${PHONE_DATA_DIR}" || exit -1
  if [ "$PRODUCTION_MODE" = false ]; then
    adb shell "mkdir -p ${KERNEL_DIR}" || exit -1
  fi
  adb push ${MODEL_DIR}/${INPUT_FILE_NAME} ${PHONE_DATA_DIR} || exit -1
  adb push bazel-bin/mace/examples/mace_run ${PHONE_DATA_DIR} || exit -1

  if [[ "${TUNING_OR_NOT}" != "0" && "$PRODUCTION_MODE" != true ]];then
    tuning_flag=1
    round=0 # only warm up
  else
    tuning_flag=0
    round=2
  fi

  adb </dev/null shell MACE_TUNING=${tuning_flag} \
    MACE_CPP_MIN_VLOG_LEVEL=$VLOG_LEVEL \
    MACE_RUN_PARAMETER_PATH=${PHONE_DATA_DIR}/mace_run.config \
    MACE_KERNEL_PATH=$KERNEL_DIR \
    ${PHONE_DATA_DIR}/mace_run \
    --input_shape="1,${IMAGE_SIZE},${IMAGE_SIZE},3"\
    --output_shape="1,${IMAGE_SIZE},${IMAGE_SIZE},2"\
    --input_file=${PHONE_DATA_DIR}/${INPUT_FILE_NAME} \
    --output_file=${PHONE_DATA_DIR}/${OUTPUT_FILE_NAME} \
    --device=OPENCL   \
    --round=$round || exit -1
}

echo "Step 1: Generate input data"
rm -rf ${MODEL_DIR}/${INPUT_FILE_NAME}
python tools/validate.py --generate_data true \
 --input_file=${MODEL_DIR}/${INPUT_FILE_NAME} \
 --input_shape="${IMAGE_SIZE},${IMAGE_SIZE},3" || exit -1

echo "Step 2: Convert tf model to mace model and optimize memory"
bazel build //mace/python/tools:tf_converter || exit -1
rm -rf ${MODEL_CODEGEN_DIR}
mkdir -p ${MODEL_CODEGEN_DIR}
bazel-bin/mace/python/tools/tf_converter --input=${TF_MODEL_FILE_PATH} \
                                         --output=${MODEL_CODEGEN_DIR}/model.cc \
                                         --input_node=${TF_INPUT_NODE} \
                                         --output_node=${TF_OUTPUT_NODE} \
                                         --data_type=DT_HALF \
                                         --runtime=gpu \
                                         --output_type=source \
                                         --template=${MACE_SOURCE_DIR}/mace/python/tools/model.template \
                                         --model_tag=${MODEL_TAG} \
                                         --obfuscate=True || exit -1

echo "Step 3: Generate version source"
rm -rf ${VERSION_SOURCE_PATH}
mkdir -p ${VERSION_SOURCE_PATH}
bash mace/tools/git/gen_version_source.sh ${VERSION_SOURCE_PATH}/version.cc

echo "Step 4: Generate encrypted opencl source"
rm -rf ${CL_CODEGEN_DIR}
mkdir -p ${CL_CODEGEN_DIR}
python mace/python/tools/encrypt_opencl_codegen.py \
  --cl_kernel_dir=./mace/kernels/opencl/cl/ --output_path=${CL_CODEGEN_DIR}/opencl_encrypt_program.cc

echo "Step 5: Run model on the phone with files"
build_and_run false

echo "Step 6: Generate OpenCL binary program and config code"
rm -rf ${CL_BIN_DIR}
adb pull ${KERNEL_DIR} ${CL_BIN_DIR}
python mace/python/tools/opencl_codegen.py \
  --cl_binary_dir=${CL_BIN_DIR} --output_path=${CL_CODEGEN_DIR}/opencl_compiled_program.cc

echo "Step 7: Generate tuning source file"
adb pull ${PHONE_DATA_DIR}/mace_run.config ${CL_BIN_DIR}
rm -rf ${TUNING_CODEGEN_DIR}
mkdir -p ${TUNING_CODEGEN_DIR}
python mace/python/tools/binary_codegen.py \
  --binary_file=${CL_BIN_DIR}/mace_run.config --output_path=${TUNING_CODEGEN_DIR}/tuning_params.cc

echo "Step 8: Run model on the phone using binary"
build_and_run true

echo "Step 9: Pull the mace run result."
rm -rf ${MODEL_DIR}/${OUTPUT_FILE_NAME}
adb </dev/null pull ${PHONE_DATA_DIR}/${OUTPUT_FILE_NAME} ${MODEL_DIR}

echo "Step 10: Validate the result"
python tools/validate.py --model_file ${TF_MODEL_FILE_PATH} \
    --input_file ${MODEL_DIR}/${INPUT_FILE_NAME} \
    --mace_out_file ${MODEL_DIR}/${OUTPUT_FILE_NAME} \
    --input_node ${TF_INPUT_NODE} \
    --output_node ${TF_OUTPUT_NODE} \
    --input_shape "${IMAGE_SIZE},${IMAGE_SIZE},3" \
    --output_shape "1,${IMAGE_SIZE},${IMAGE_SIZE},2"
