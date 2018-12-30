#!/bin/bash
#usage: submit.sh $EXPERIMENT_NAME
ROOT_DIR="/project/RDS-FEI-NLH-RW/work/pprli/"
NAME=$1
COMMAND="python src/train.py --name $NAME"
EXPERIMENT_DIR="$ROOT_DIR/$NAME"
PBS_PATH="$EXPERIMENT_DIR/$NAME.pbs"
echo "$COMMAND"
echo "$EXPERIMENT_DIR"
git add --all
git commit -m "Experiment $COMMAND"
mkdir -p $EXPERIMENT_DIR
cp scripts/template.pbs $PBS_PATH
echo $COMMAND >> $PBS_PATH
echo "#$(git rev-parse HEAD)" >> $PBS_PATH
cd $EXPERIMENT_DIR
jobid=$(qsub $PBS_PATH)
echo $jobid
