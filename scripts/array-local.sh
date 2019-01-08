#!/bin/bash
#usage: finetune.sh $EXPERIMENT_NAME
MAX_JOBS=5
ROOT_DIR="/project/RDS-FEI-NLH-RW/work/pprli/"
CONFIG_DIR="$ROOT_DIR/configs/"
NAME=$1
ADD_NAME=${2:-$(date '+%Y-%m-%d-%H-%M-%S')}
echo $ADD_NAME
EXPERIMENT_DIR="$ROOT_DIR/experiments/tune-${NAME}-${ADD_NAME}"
mkdir -p $EXPERIMENT_DIR
#generate config file
COMMAND="python $CONFIG_DIR/tune_config.py --type $NAME --name $ADD_NAME"
$COMMAND
CONFIG_LIST_PATH="$EXPERIMENT_DIR/configs.list"
num_files=`cat ${CONFIG_LIST_PATH} | wc -l | tr -d ' '`
num_jobs=$(($num_files>${MAX_JOBS}?${MAX_JOBS}:$num_files))
echo $num_files

for ((i=1;i<=${num_files};i+=${num_jobs})); do
    config_name=`sed "${i}q;d" ${CONFIG_LIST_PATH}`
    python src/train.py --name ${config_name} --hpc --debug
done || exit 1

