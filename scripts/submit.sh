#!/bin/bash
#usage: submit.sh $EXPERIMENT_NAME
ROOT_DIR="/project/RDS-FEI-NLH-RW/work/pprli/"
CONFIG_DIR="$ROOT_DIR/configs/"
NAME=$1
NEW_NAME=$NAME-$(date '+%Y-%m-%d-%H-%M-%S')
COMMAND="python src/train.py --name $NEW_NAME --hpc"
EXPERIMENT_DIR="$ROOT_DIR/experiments/$NEW_NAME"
PBS_PATH="$EXPERIMENT_DIR/$NEW_NAME.pbs"

echo "$COMMAND"
echo "$EXPERIMENT_DIR"
mkdir -p $EXPERIMENT_DIR
cp $CONFIG_DIR/$NAME.py $EXPERIMENT_DIR/$NEW_NAME.py
cp scripts/template.pbs $PBS_PATH
echo $COMMAND >> $PBS_PATH
git add --all
git commit -m "Experiment $COMMAND"
echo "#$(git rev-parse HEAD)" >> $PBS_PATH
exit 1
cd $EXPERIMENT_DIR
jobid=$(qsub $PBS_PATH)
echo $jobid
