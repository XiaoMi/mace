#!/bin/bash

Usage() {
  echo "Usage: bash tools/validate_tools.sh target_soc model_output_dir generate_data_or_not"
}

if [ $# -lt 3 ]; then
  Usage
  exit 1
fi

CURRENT_DIR=`dirname $0`
source ${CURRENT_DIR}/env.sh

TARGET_SOC=$1
MODEL_OUTPUT_DIR=$2
GENERATE_DATA_OR_NOT=$3

DEVICE_ID=`echo_device_id_by_soc $TARGET_SOC`

IFS=',' read -r -a INPUT_NAMES <<< "${INPUT_NODES}"
IFS=',' read -r -a OUTPUT_NAMES <<< "${OUTPUT_NODES}"

echo $MODEL_OUTPUT_DIR
if [ "$GENERATE_DATA_OR_NOT" = 1 ]; then
  for NAME in "${INPUT_NAMES[@]}";do
    FORMATTED_NAME=$(sed s/[^[:alnum:]]/_/g <<< ${NAME})
    rm -rf ${MODEL_OUTPUT_DIR}/${INPUT_FILE_NAME}_${FORMATTED_NAME}
  done
  python -u tools/generate_data.py --input_node=${INPUT_NODES} \
    --input_file=${MODEL_OUTPUT_DIR}/${INPUT_FILE_NAME} \
    --input_shape="${INPUT_SHAPES}" || exit 1
  exit 0
fi

if [ "$PLATFORM" == "tensorflow" ];then
  if [[ x"$TARGET_ABI" != x"host" ]]; then
    for NAME in "${OUTPUT_NAMES[@]}";do
      FORMATTED_NAME=$(sed s/[^[:alnum:]]/_/g <<< ${NAME})
      rm -rf ${MODEL_OUTPUT_DIR}/${OUTPUT_FILE_NAME}_${FORMATTED_NAME}
      adb -s $DEVICE_ID pull ${PHONE_DATA_DIR}/${OUTPUT_FILE_NAME}_${FORMATTED_NAME} ${MODEL_OUTPUT_DIR} > /dev/null || exit 1
    done
  fi
  python -u tools/validate.py --platform=tensorflow \
      --model_file ${MODEL_FILE_PATH} \
      --input_file ${MODEL_OUTPUT_DIR}/${INPUT_FILE_NAME} \
      --mace_out_file ${MODEL_OUTPUT_DIR}/${OUTPUT_FILE_NAME} \
      --mace_runtime ${RUNTIME} \
      --input_node ${INPUT_NODES} \
      --output_node ${OUTPUT_NODES} \
      --input_shape ${INPUT_SHAPES} \
      --output_shape ${OUTPUT_SHAPES} || exit 1

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

  if [[ x"$TARGET_ABI" != x"host" ]]; then
    for NAME in "${OUTPUT_NAMES[@]}";do
      FORMATTED_NAME=$(sed s/[^[:alnum:]]/_/g <<< ${NAME})
      rm -rf ${MODEL_OUTPUT_DIR}/${OUTPUT_FILE_NAME}_${FORMATTED_NAME}
      adb -s $DEVICE_ID pull ${PHONE_DATA_DIR}/${OUTPUT_FILE_NAME}_${FORMATTED_NAME} ${MODEL_OUTPUT_DIR} > /dev/null || exit 1
    done
  fi
  for NAME in "${OUTPUT_NAMES[@]}";do
    FORMATTED_NAME=$(sed s/[^[:alnum:]]/_/g <<< ${NAME})
    docker cp ${MODEL_OUTPUT_DIR}/${OUTPUT_FILE_NAME}_${FORMATTED_NAME} ${CONTAINER_NAME}:/mace
  done

  MODEL_FILE_NAME=$(basename ${MODEL_FILE_PATH})
  WEIGHT_FILE_NAME=$(basename ${WEIGHT_FILE_PATH})
  docker cp tools/validate.py ${CONTAINER_NAME}:/mace
  docker cp ${MODEL_FILE_PATH} ${CONTAINER_NAME}:/mace
  docker cp ${WEIGHT_FILE_PATH} ${CONTAINER_NAME}:/mace
  docker exec -it ${CONTAINER_NAME} python -u /mace/validate.py \
    --platform=caffe \
    --model_file /mace/${MODEL_FILE_NAME} \
    --weight_file /mace/${WEIGHT_FILE_NAME} \
    --input_file /mace/${INPUT_FILE_NAME} \
    --mace_out_file /mace/${OUTPUT_FILE_NAME} \
    --mace_runtime ${RUNTIME} \
    --input_node ${INPUT_NODES} \
    --output_node ${OUTPUT_NODES} \
    --input_shape ${INPUT_SHAPES} \
    --output_shape ${OUTPUT_SHAPES} || exit 1

fi
