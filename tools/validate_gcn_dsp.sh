#!/bin/bash
# Must run at root dir of mace project.
set +x
Usage() {
  echo 'Usage: bash tools/validate_gcn.sh tools/gcn.config tf_model_path image_size [tuning]'
}

if [ $# -lt 2 ];then
  Usage
  exit -1
fi

source $1

VLOG_LEVEL=0
TF_MODEL_FILE_PATH=$2
MODEL_DIR=$(dirname ${TF_MODEL_FILE_PATH})
MACE_SOURCE_DIR=`/bin/pwd`
MACE_MODEL_NAME='mace_model.pb'
INPUT_FILE_NAME='model_input'
OUTPUT_FILE_NAME='gcn.out'
OUTPUT_LIST_FILE='gcn.list'
PHONE_DATA_DIR="/data/local/tmp/${MACE_MODEL_NAME}"
KERNEL_DIR="${PHONE_DATA_DIR}/cl/"
IMAGE_SIZE=$3
MODEL_TAG=GCN${IMAGE_SIZE}
CODEGEN_DIR=${MACE_SOURCE_DIR}/mace/codegen
MODEL_CODEGEN_DIR=${CODEGEN_DIR}/models/gcn-$IMAGE_SIZE
VERSION_SOURCE_PATH=${CODEGEN_DIR}/version

build_and_run()
{
  bazel build -c opt --strip always mace/examples:mace_run \
    --crosstool_top=//external:android/crosstool \
    --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
    --cpu=armeabi-v7a \
    --copt=-DMACE_MODEL_FUNCTION=Create${MODEL_TAG}

  adb shell "mkdir -p ${PHONE_DATA_DIR}"
  adb push ${MODEL_DIR}/${INPUT_FILE_NAME} ${PHONE_DATA_DIR}
  adb push bazel-bin/mace/examples/mace_run ${PHONE_DATA_DIR}

  adb </dev/null shell \
    MACE_CPP_MIN_VLOG_LEVEL=$VLOG_LEVEL \
    MACE_RUN_PARAMETER_PATH=${PHONE_DATA_DIR}/mace_run.config \
    ${PHONE_DATA_DIR}/mace_run \
    --input_shape="1,${IMAGE_SIZE},${IMAGE_SIZE},3"\
    --output_shape="1,${IMAGE_SIZE},${IMAGE_SIZE},2"\
    --input_file=${PHONE_DATA_DIR}/${INPUT_FILE_NAME} \
    --output_file=${PHONE_DATA_DIR}/${OUTPUT_FILE_NAME} \
    --device=HEXAGON   \
    --round=$round
}

echo "Step 1: Generate input data"
rm -rf ${MODEL_DIR}/${INPUT_FILE_NAME}
python tools/validate.py --generate_data true \
 --input_file=${MODEL_DIR}/${INPUT_FILE_NAME} \
 --input_shape="${IMAGE_SIZE},${IMAGE_SIZE},3"

echo "Step 2: Convert tf model to mace model and optimize memory"
bazel build //mace/python/tools:tf_converter
rm -rf ${MODEL_CODEGEN_DIR}
mkdir -p ${MODEL_CODEGEN_DIR}
bazel-bin/mace/python/tools/tf_converter --input=${TF_MODEL_FILE_PATH} \
                                         --output=${MODEL_CODEGEN_DIR}/mace_gcn${IMAGE_SIZE}.cc \
                                         --input_node=${TF_INPUT_NODE} \
                                         --output_node=${TF_OUTPUT_NODE} \
                                         --data_type=DT_UINT8 \
                                         --runtime=dsp \
                                         --output_type=source \
                                         --template=${MACE_SOURCE_DIR}/mace/python/tools/model.template \
                                         --model_tag=${MODEL_TAG} \
                                         --confuse=True

echo "Step 3: Generate version source"
rm -rf ${VERSION_SOURCE_PATH}
mkdir -p ${VERSION_SOURCE_PATH}
bash mace/tools/git/gen_version_source.sh ${VERSION_SOURCE_PATH}/version.cc

echo "Step 4: Run model on the phone with files"
build_and_run

echo "Step 5: Pull the mace run result."
rm -rf ${MODEL_DIR}/${OUTPUT_FILE_NAME}
adb </dev/null pull ${PHONE_DATA_DIR}/${OUTPUT_FILE_NAME} ${MODEL_DIR}

echo "Step 6: Validate the result"
python tools/validate.py --model_file ${TF_MODEL_FILE_PATH} \
    --input_file ${MODEL_DIR}/${INPUT_FILE_NAME} \
    --mace_out_file ${MODEL_DIR}/${OUTPUT_FILE_NAME} \
    --input_node ${TF_INPUT_NODE} \
    --output_node ${TF_OUTPUT_NODE} \
    --input_shape "${IMAGE_SIZE},${IMAGE_SIZE},3" \
    --output_shape "1,${IMAGE_SIZE},${IMAGE_SIZE},2"