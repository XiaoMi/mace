#!/bin/bash
# Must run at root dir of mace project.
set -e

Usage() {
  echo 'Usage: bash tools/validate_icnet.sh tf_model_file'
}

if [ $# != 1 ];then
  Usage
  exit -1
fi

TF_MODEL_FILE_PATH=$1
MODEL_DIR=$(dirname ${TF_MODEL_FILE_PATH})
MACE_MODEL_NAME='mace_model.pb'
TF_INPUT_FILE_NAME='tf_model_input'
MACE_INPUT_FILE_NAME='mace_model_input'
OUTPUT_FILE_NAME='icnet.out'
PHONE_DATA_DIR="/data/local/tmp/${MACE_MODEL_NAME}"
KERNEL_DIR="${PHONE_DATA_DIR}/cl/"

# Step 1: convert tf model to mace model
echo "Step 1: convert tf model to mace model"
bazel build //mace/python/tools:tf_converter
bazel-bin/mace/python/tools/tf_converter --input=${TF_MODEL_FILE_PATH} --output=${MODEL_DIR}/${MACE_MODEL_NAME}

# Step 2: Generate input data
echo "Step 2: Generate input data"
python tools/validate_icnet.py --generate_data true --random_seed 1 \
 --mace_input_file ${MODEL_DIR}/${MACE_INPUT_FILE_NAME} --tf_input_file ${MODEL_DIR}/${TF_INPUT_FILE_NAME}

# Step 3: Run model on the phone
echo "Step 3: Run model on the phone"
bazel build -c opt --strip always mace/examples:mace_run  \
    --crosstool_top=//external:android/crosstool \
    --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
    --cpu=arm64-v8a

adb shell "mkdir -p ${PHONE_DATA_DIR}"
adb shell "mkdir -p ${KERNEL_DIR}"
adb push mace/kernels/opencl/cl/* ${KERNEL_DIR}
adb push ${MODEL_DIR}/${MACE_MODEL_NAME} ${PHONE_DATA_DIR}
adb push ${MODEL_DIR}/${MACE_INPUT_FILE_NAME} ${PHONE_DATA_DIR}
adb push bazel-bin/mace/examples/mace_run ${PHONE_DATA_DIR}

num_threads=${1:-1}

adb shell MACE_RUN_PARAMETER_PATH=${PHONE_DATA_DIR}/mace_run.config \
          MACE_KERNEL_PATH=$KERNEL_DIR \
          OMP_NUM_THREADS=$num_threads \
          ${PHONE_DATA_DIR}/mace_run \
            --input_shape=1,3,480,480\
            --input_file=${PHONE_DATA_DIR}/${MACE_INPUT_FILE_NAME} \
            --output_file=${PHONE_DATA_DIR}/${OUTPUT_FILE_NAME} \
            --device=OPENCL

# Step 4: pull the mace run result.
echo "Step 4: pull the mace run result."
adb pull ${PHONE_DATA_DIR}/${OUTPUT_FILE_NAME} ${MODEL_DIR}

# Step 5: validate the result
echo "Step 5: validate the result"
python tools/validate_icnet.py --model_file ${TF_MODEL_FILE_PATH} \
  --tf_input_file ${MODEL_DIR}/${TF_INPUT_FILE_NAME} \
  --mace_out_file ${MODEL_DIR}/${OUTPUT_FILE_NAME}


