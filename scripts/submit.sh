#!/bin/bash
#usage: submit.sh $EXPERIMENT_NAME $YOUR_COMMAND
ROOT_DIR="/project/RDS-FEI-NLH-RW/work/pprli/"
NAME=$1-$(date '+%Y-%m-%d-%H-%M-%S')
EXPERIMENT_DIR=$ROOT_DIR/experiments/$NAME
PBS_PATH=$ROOT_DIR/hpc_scripts/$NAME.pbs
COMMAND=${@:2}
echo "$COMMAND"
$COMMAND --debug
if [ "$?" -gt 0 ]; then
    echo "Fail because command don't pass debug test"
    exit 1
fi
echo "passed test. submit job now"
mkdir -p $EXPERIMENT_DIR
cp scripts/template.pbs $PBS_PATH
echo "$COMMAND --name $NAME" >> $PBS_PATH
git add --all
git commit -m "Experiment $NAME $COMMAND"
git push
echo "#$(git rev-parse HEAD)" >> $PBS_PATH
cd $EXPERIMENT_DIR
jobid=$(qsub $PBS_PATH)
echo $jobid
<<COMMENT
#Todo: figure out how to get expected file name
expected=$NAME.pbs.e$jobid
while [ ! -f "$expected" ]
do
    inotifywait -qqt 2 -e create -e moved_to "$(dirname $expected)"
done
COMMENT
