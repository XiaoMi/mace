#!/bin/bash
# Must run at root dir of mace project.

Usage() {
  echo 'Usage: bash tools/validate_gcn.sh tf_model_file'
}

if [ $# != 1 ];then
  Usage
  exit -1
fi

TF_MODEL_FILE_PATH=$1
MODEL_DIR=$(dirname ${TF_MODEL_FILE_PATH})
MACE_MODEL_NAME='mace_model.pb'
INPUT_FILE_NAME='model_input'
OUTPUT_FILE_NAME='gcn.out'
OUTPUT_LIST_FILE='gcn.list'
PHONE_DATA_DIR="/data/local/tmp/${MACE_MODEL_NAME}"
KERNEL_DIR="${PHONE_DATA_DIR}/cl/"

# Step 1: Generate input data
echo "Step 1: Generate input data"
python tools/validate.py --generate_data true --random_seed 1 \
 --input_file=${MODEL_DIR}/${INPUT_FILE_NAME} \
 --input_shape=512,512,3

# Step 2: convert tf model to mace model
echo "Step 2: convert tf model to mace model"
bazel build //mace/python/tools:tf_converter
bazel-bin/mace/python/tools/tf_converter --input=${TF_MODEL_FILE_PATH} \
                                         --output=${MODEL_DIR}/${MACE_MODEL_NAME} \
                                         --input_node=input \
                                         --output_node=GCN/br_result_2/fcn_br \
                                         --data_type=DT_HALF\
                                         --runtime=gpu


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
adb push ${MODEL_DIR}/${INPUT_FILE_NAME} ${PHONE_DATA_DIR}
adb push bazel-bin/mace/examples/mace_run ${PHONE_DATA_DIR}

num_threads=${1:-4}

adb </dev/null shell MACE_RUN_PARAMETER_PATH=${PHONE_DATA_DIR}/mace_run.config \
        MACE_KERNEL_PATH=$KERNEL_DIR \
        OMP_NUM_THREADS=$num_threads \
        ${PHONE_DATA_DIR}/mace_run \
          --model=${PHONE_DATA_DIR}/${MACE_MODEL_NAME} \
          --input=mace_input_node \
          --output=mace_output_node \
          --input_shape=1,512,512,3\
          --input_file=${PHONE_DATA_DIR}/${INPUT_FILE_NAME} \
          --output_file=${PHONE_DATA_DIR}/${OUTPUT_FILE_NAME} \
          --device=OPENCL

# Step 4: pull the mace run result.
echo "Step 4: pull the mace run result."
rm -rf ${MODEL_DIR}/${OUTPUT_FILE_NAME}
adb </dev/null pull ${PHONE_DATA_DIR}/${OUTPUT_FILE_NAME} ${MODEL_DIR}

# Step 5: validate the result
echo "Step 5: validate the result"
python tools/validate.py --model_file ${TF_MODEL_FILE_PATH} \
    --input_file ${MODEL_DIR}/${INPUT_FILE_NAME} \
    --mace_out_file ${MODEL_DIR}/${OUTPUT_FILE_NAME} \
    --input_node input \
    --output_node GCN/br_result_2/fcn_br\
    --output_shape 1,512,512,2
