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
VERSION_SOURCE_PATH=${CODEGEN_DIR}/version
CL_CODEGEN_DIR=${CODEGEN_DIR}/opencl
CL_BIN_DIR=${CODEGEN_DIR}/opencl_bin
TUNING_CODEGEN_DIR=${CODEGEN_DIR}/tuning

build_and_run()
{
  bazel build -c opt --strip always mace/examples:mace_run \
    --crosstool_top=//external:android/crosstool \
    --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
    --cpu=armeabi-v7a \
    --copt="-std=c++11" \
    --copt="-D_GLIBCXX_USE_C99_MATH_TR1" \
    --copt="-Werror=return-type" \
    --copt="-DMACE_MODEL_TAG=${MODEL_TAG}" \
    --define hexagon=true --define production=true || exit -1

  adb shell "mkdir -p ${PHONE_DATA_DIR}" || exit -1
  adb push ${MODEL_DIR}/${INPUT_FILE_NAME} ${PHONE_DATA_DIR} || exit -1
  adb push bazel-bin/mace/examples/mace_run ${PHONE_DATA_DIR} || exit -1
  adb push mace/core/runtime/hexagon/libhexagon_controller.so ${PHONE_DATA_DIR} || exit -1

  adb </dev/null shell \
    LD_LIBRARY_PATH=${PHONE_DATA_DIR} \
    MACE_CPP_MIN_VLOG_LEVEL=$VLOG_LEVEL \
    MACE_RUN_PARAMETER_PATH=${PHONE_DATA_DIR}/mace_run.config \
    ${PHONE_DATA_DIR}/mace_run \
    --input_shape="1,${IMAGE_SIZE},${IMAGE_SIZE},3"\
    --output_shape="1,${IMAGE_SIZE},${IMAGE_SIZE},2"\
    --input_file=${PHONE_DATA_DIR}/${INPUT_FILE_NAME} \
    --output_file=${PHONE_DATA_DIR}/${OUTPUT_FILE_NAME} \
    --device=HEXAGON \
    --round=2 || exit -1
}

echo "Step 1: Generate input data"
rm -rf ${MODEL_DIR}/${INPUT_FILE_NAME}
python tools/validate.py --generate_data true \
 --input_file=${MODEL_DIR}/${INPUT_FILE_NAME} \
 --input_shape="${IMAGE_SIZE},${IMAGE_SIZE},3" || exit -1

echo "Step 2: Convert tf model to mace model and optimize memory"
bazel build //mace/python/tools:tf_converter
rm -rf ${MODEL_CODEGEN_DIR}
mkdir -p ${MODEL_CODEGEN_DIR}
bazel-bin/mace/python/tools/tf_converter --input=${TF_MODEL_FILE_PATH} \
                                         --output=${MODEL_CODEGEN_DIR}/mace_gcn${IMAGE_SIZE}.cc \
                                         --input_node=${TF_INPUT_NODE} \
                                         --output_node=${TF_OUTPUT_BR_NODE} \
                                         --data_type=DT_UINT8 \
                                         --runtime=dsp \
                                         --output_type=source \
                                         --template=${MACE_SOURCE_DIR}/mace/python/tools/model.template \
                                         --model_tag=${MODEL_TAG} \
                                         --obfuscate=True || exit -1

echo "Step 3: Generate version source"
rm -rf ${VERSION_SOURCE_PATH}
mkdir -p ${VERSION_SOURCE_PATH}
bash mace/tools/git/gen_version_source.sh ${VERSION_SOURCE_PATH}/version.cc

echo "Step 4: Generate OpenCL binary program and config code"
rm -rf ${CL_BIN_DIR}
mkdir -p ${CL_BIN_DIR}
python mace/python/tools/opencl_codegen.py \
  --cl_binary_dir=${CL_BIN_DIR} --output_path=${CL_CODEGEN_DIR}/opencl_compiled_program.cc

echo "Step 5: Generate tuning source file"
rm -rf ${TUNING_CODEGEN_DIR}
mkdir -p ${TUNING_CODEGEN_DIR}
python mace/python/tools/binary_codegen.py \
  --binary_file=${CL_BIN_DIR}/mace_run.config --output_path=${TUNING_CODEGEN_DIR}/tuning_params.cc

echo "Step 6: Run model on the phone with files"
build_and_run

echo "Step 7: Pull the mace run result."
rm -rf ${MODEL_DIR}/${OUTPUT_FILE_NAME}
adb </dev/null pull ${PHONE_DATA_DIR}/${OUTPUT_FILE_NAME} ${MODEL_DIR}

echo "Step 8: Validate the result"
python tools/validate.py --model_file ${TF_MODEL_FILE_PATH} \
    --input_file ${MODEL_DIR}/${INPUT_FILE_NAME} \
    --mace_out_file ${MODEL_DIR}/${OUTPUT_FILE_NAME} \
    --input_node ${TF_INPUT_NODE} \
    --output_node ${TF_OUTPUT_BR_NODE} \
    --input_shape "${IMAGE_SIZE},${IMAGE_SIZE},3" \
    --output_shape "1,${IMAGE_SIZE},${IMAGE_SIZE},2"
