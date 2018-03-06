#!/bin/bash

Usage() {
  echo "Usage: bash tools/validate_tools.sh model_output_dir generate_data_or_not"
}

if [ $# -lt 2 ]; then
  Usage
  exit 1
fi

CURRENT_DIR=`dirname $0`
source ${CURRENT_DIR}/env.sh

MODEL_OUTPUT_DIR=$1
GENERATE_DATA_OR_NOT=$2

IFS=',' read -r -a INPUT_NAMES <<< "${INPUT_NODE}"
IFS=',' read -r -a OUTPUT_NAMES <<< "${OUTPUT_NODE}"

echo $MODEL_OUTPUT_DIR
if [ "$GENERATE_DATA_OR_NOT" = 1 ]; then
  for NAME in "${INPUT_NAMES[@]}";do
    FORMATTED_NAME=$(sed s/[^[:alnum:]]/_/g <<< ${NAME})
    rm -rf ${MODEL_OUTPUT_DIR}/${INPUT_FILE_NAME}_${FORMATTED_NAME}
  done
  python tools/generate_data.py --input_node=${INPUT_NODE} \
    --input_file=${MODEL_OUTPUT_DIR}/${INPUT_FILE_NAME} \
    --input_shape="${INPUT_SHAPE}" || exit 1
  exit 0
fi

if [ "$PLATFORM" == "tensorflow" ];then
  if [[ x"$TARGET_ABI" -ne x"host" ]]; then
    for NAME in "${OUTPUT_NAMES[@]}";do
      FORMATTED_NAME=$(sed s/[^[:alnum:]]/_/g <<< ${NAME})
      rm -rf ${MODEL_OUTPUT_DIR}/${OUTPUT_FILE_NAME}_${FORMATTED_NAME}
      adb </dev/null pull ${PHONE_DATA_DIR}/${OUTPUT_FILE_NAME}_${FORMATTED_NAME} ${MODEL_OUTPUT_DIR}
    done
  fi
  python tools/validate.py --platform=tensorflow \
      --model_file ${MODEL_FILE_PATH} \
      --input_file ${MODEL_OUTPUT_DIR}/${INPUT_FILE_NAME} \
      --mace_out_file ${MODEL_OUTPUT_DIR}/${OUTPUT_FILE_NAME} \
      --mace_runtime ${RUNTIME} \
      --input_node ${INPUT_NODE} \
      --output_node ${OUTPUT_NODE} \
      --input_shape ${INPUT_SHAPE} \
      --output_shape ${OUTPUT_SHAPE} || exit 1

elif [ "$PLATFORM" == "caffe" ];then
  IMAGE_NAME=mace-caffe:latest
  CONTAINER_NAME=mace_caffe_validator
  RES_FILE=validation.result

  if [[ "$(docker images -q mace-caffe:latest 2> /dev/null)" == "" ]]; then
    echo "Build caffe docker"
    docker build -t ${IMAGE_NAME} docker/caffe || exit 1
  fi

  if [ ! "$(docker ps -qa -f name=${CONTAINER_NAME})" ]; then
    echo "Run caffe container"
    docker run -d -it --name ${CONTAINER_NAME} ${IMAGE_NAME} /bin/bash || exit 1
  fi

  if [ "$(docker inspect -f {{.State.Running}} ${CONTAINER_NAME})" == "false" ];then
    echo "Start caffe container"
    docker start ${CONTAINER_NAME}
  fi

  for NAME in "${INPUT_NAMES[@]}";do
    FORMATTED_NAME=$(sed s/[^[:alnum:]]/_/g <<< ${NAME})
    docker cp ${MODEL_OUTPUT_DIR}/${INPUT_FILE_NAME}_${FORMATTED_NAME} ${CONTAINER_NAME}:/mace
  done

  if [[ x"$TARGET_ABI" -ne x"host" ]]; then
    for NAME in "${OUTPUT_NAMES[@]}";do
      FORMATTED_NAME=$(sed s/[^[:alnum:]]/_/g <<< ${NAME})
      rm -rf ${MODEL_OUTPUT_DIR}/${OUTPUT_FILE_NAME}_${FORMATTED_NAME}
      adb </dev/null pull ${PHONE_DATA_DIR}/${OUTPUT_FILE_NAME}_${FORMATTED_NAME} ${MODEL_OUTPUT_DIR}
      docker cp ${MODEL_OUTPUT_DIR}/${OUTPUT_FILE_NAME}_${FORMATTED_NAME} ${CONTAINER_NAME}:/mace
    done
  fi

  MODEL_FILE_NAME=$(basename ${MODEL_FILE_PATH})
  WEIGHT_FILE_NAME=$(basename ${WEIGHT_FILE_PATH})
  docker cp tools/validate.py ${CONTAINER_NAME}:/mace
  docker cp ${MODEL_FILE_PATH} ${CONTAINER_NAME}:/mace
  docker cp ${WEIGHT_FILE_PATH} ${CONTAINER_NAME}:/mace
  docker exec -it ${CONTAINER_NAME} python /mace/validate.py  --platform=caffe \
    --model_file /mace/${MODEL_FILE_NAME} \
    --weight_file /mace/${WEIGHT_FILE_NAME} \
    --input_file /mace/${INPUT_FILE_NAME} \
    --mace_out_file /mace/${OUTPUT_FILE_NAME} \
    --mace_runtime ${RUNTIME} \
    --input_node ${INPUT_NODE} \
    --output_node ${OUTPUT_NODE} \
    --input_shape ${INPUT_SHAPE} \
    --output_shape ${OUTPUT_SHAPE}

fi
