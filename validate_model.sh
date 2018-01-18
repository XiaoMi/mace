#!/bin/bash
# Must run at root dir of mace project.
set +x
Usage() {
  echo 'Usage: bash tools/validate_gcn.sh tools/gcn.config tf_model_path model_tag image_size runtime[gpu/dsp] [tuning]'
}

if [ $# -lt 5 ];then
  Usage
  exit -1
fi

source $1

TF_MODEL_FILE_PATH=$2
MODEL_TAG=$3
IMAGE_SIZE=$4
RUNTIME=$5
TUNING_OR_NOT=${6:-0}

if [ x"$RUNTIME" = x"dsp" ]; then
  DATA_TYPE="DT_UINT8"
  DEVICE_TYPE="HEXAGON"
  TF_OUTPUT_NODE=${TF_OUTPUT_BR_NODE}
elif [ x"$RUNTIME" = x"gpu" ]; then
  DATA_TYPE="DT_HALF"
  DEVICE_TYPE="OPENCL"
else
  Usage
  exit -1
fi

VLOG_LEVEL=0
MODEL_DIR=$(dirname ${TF_MODEL_FILE_PATH})
LIBMACE_SOURCE_DIR=`/bin/pwd`
INPUT_FILE_NAME='model_input'
OUTPUT_FILE_NAME='model.out'
OUTPUT_LIST_FILE='model.list'
PHONE_DATA_DIR="/data/local/tmp/mace_run"
KERNEL_DIR="${PHONE_DATA_DIR}/cl/"
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

  if [[ "${TUNING_OR_NOT}" != "0" && "$PRODUCTION_MODE" != true ]];then
    tuning_flag=1
    round=0 # only warm up
  else
    tuning_flag=0
    round=2
  fi

  if [ x"$RUNTIME" = x"dsp" ]; then
    HEXAGON_MODE_BUILD_FLAGS="--define hexagon=true"
  fi

  bazel build --verbose_failures -c opt --strip always examples:mace_run \
    --crosstool_top=//external:android/crosstool \
    --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
    --cpu=armeabi-v7a \
    --copt="-std=c++11" \
    --copt="-D_GLIBCXX_USE_C99_MATH_TR1" \
    --copt="-Werror=return-type" \
    --copt="-DMACE_MODEL_TAG=${MODEL_TAG}" \
    --copt="-DMACE_OBFUSCATE_LITERALS" \
    $PRODUCTION_MODE_BUILD_FLAGS \
    $HEXAGON_MODE_BUILD_FLAGS || exit -1

  adb shell "mkdir -p ${PHONE_DATA_DIR}" || exit -1
  if [ "$PRODUCTION_MODE" = false ]; then
    adb shell "mkdir -p ${KERNEL_DIR}" || exit -1
  fi
  adb push ${MODEL_DIR}/${INPUT_FILE_NAME} ${PHONE_DATA_DIR} || exit -1
  adb push bazel-bin/examples/mace_run ${PHONE_DATA_DIR} || exit -1
  adb push lib/hexagon/libhexagon_controller.so ${PHONE_DATA_DIR} || exit 0

  adb </dev/null shell \
    LD_LIBRARY_PATH=${PHONE_DATA_DIR} \
    MACE_TUNING=${tuning_flag} \
    MACE_CPP_MIN_VLOG_LEVEL=$VLOG_LEVEL \
    MACE_RUN_PARAMETER_PATH=${PHONE_DATA_DIR}/mace_run.config \
    MACE_KERNEL_PATH=$KERNEL_DIR \
    ${PHONE_DATA_DIR}/mace_run \
    --input_shape="1,${IMAGE_SIZE},${IMAGE_SIZE},3"\
    --output_shape="1,${IMAGE_SIZE},${IMAGE_SIZE},2"\
    --input_file=${PHONE_DATA_DIR}/${INPUT_FILE_NAME} \
    --output_file=${PHONE_DATA_DIR}/${OUTPUT_FILE_NAME} \
    --device=${DEVICE_TYPE}   \
    --round=$round || exit -1
}

echo "Step 1: Generate input data"
rm -rf ${MODEL_DIR}/${INPUT_FILE_NAME}
python tools/validate.py --generate_data true \
 --input_file=${MODEL_DIR}/${INPUT_FILE_NAME} \
 --input_shape="${IMAGE_SIZE},${IMAGE_SIZE},3" || exit -1

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
                                        --obfuscate=True || exit -1

echo "Step 3: Run model on the phone with files"
build_and_run false

echo "Step 4: Generate OpenCL binary program and config code"
rm -rf ${CL_BIN_DIR}
rm -rf ${CL_CODEGEN_DIR}
mkdir -p ${CL_BIN_DIR}
mkdir -p ${CL_CODEGEN_DIR}
adb pull ${KERNEL_DIR} ${CL_BIN_DIR}
python lib/python/tools/opencl_codegen.py \
  --cl_binary_dir=${CL_BIN_DIR} --output_path=${CL_CODEGEN_DIR}/opencl_compiled_program.cc

echo "Step 5: Generate tuning source file"
adb pull ${PHONE_DATA_DIR}/mace_run.config ${CL_BIN_DIR}
rm -rf ${TUNING_CODEGEN_DIR}
mkdir -p ${TUNING_CODEGEN_DIR}
python lib/python/tools/binary_codegen.py \
  --binary_file=${CL_BIN_DIR}/mace_run.config --output_path=${TUNING_CODEGEN_DIR}/tuning_params.cc

echo "Step 6: Run model on the phone using binary"
build_and_run true

echo "Step 7: Pull the mace run result."
rm -rf ${MODEL_DIR}/${OUTPUT_FILE_NAME}
adb </dev/null pull ${PHONE_DATA_DIR}/${OUTPUT_FILE_NAME} ${MODEL_DIR}

echo "Step 8: Validate the result"
python tools/validate.py --model_file ${TF_MODEL_FILE_PATH} \
    --input_file ${MODEL_DIR}/${INPUT_FILE_NAME} \
    --mace_out_file ${MODEL_DIR}/${OUTPUT_FILE_NAME} \
    --input_node ${TF_INPUT_NODE} \
    --output_node ${TF_OUTPUT_NODE} \
    --input_shape "${IMAGE_SIZE},${IMAGE_SIZE},3" \
    --output_shape "1,${IMAGE_SIZE},${IMAGE_SIZE},2"
