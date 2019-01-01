#!/bin/bash
#usage: submit.sh $EXPERIMENT_NAME
ROOT_DIR="/project/RDS-FEI-NLH-RW/work/pprli/"
CONFIG_DIR="$ROOT_DIR/configs/"
NAME=$1
NEW_NAME=$NAME-$(date '+%Y-%m-%d-%H-%M-%S')
EXPERIMENT_DIR="$ROOT_DIR/experiments/$NEW_NAME"
PBS_PATH="$EXPERIMENT_DIR/$NEW_NAME.pbs"

COMMAND="python src/train.py --name $NEW_NAME --hpc"
MAKE_CONFIG_COMMAND="python configs/$NAME.py --name $NEW_NAME"

echo "MAKE_CONFIG_COMMAND=$MAKE_CONFIG_COMMAND"
echo "COMMAND=$COMMAND"
mkdir -p $EXPERIMENT_DIR
cp scripts/template.pbs $PBS_PATH
echo $MAKE_CONFIG_COMMAND >> $PBS_PATH
echo $COMMAND >> $PBS_PATH
git add --all
git commit -m "Experiment $COMMAND"
echo "#$(git rev-parse HEAD)" >> $PBS_PATH
$MAKE_CONFIG_COMMAND && \
$COMMAND
exit 1
cd $EXPERIMENT_DIR
jobid=$(qsub $PBS_PATH)
echo $jobid
