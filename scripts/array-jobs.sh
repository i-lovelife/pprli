#!/bin/bash
#usage: finetune.sh $EXPERIMENT_NAME
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
echo $num_files

PBS_PATH="${EXPERIMENT_DIR}/array_job.pbs"
cat > $PBS_PATH << EOF
#!/bin/bash
#PBS -P NLH
#PBS -q alloc-dt
#PBS -l select=1:ncpus=8:ngpus=1:mpiprocs=8:mem=45gb
#PBS -l walltime=200:00:00
#PBS -J 1-${num_files}
 
cd ${ROOT_DIR}
module load cuda/9.1.85 openmpi-gcc/3.0.0-cuda python/3.6.5
source pprli_env/bin/activate

config_name=\`sed "\${PBS_ARRAY_INDEX}q;d" ${CONFIG_LIST_PATH}\`
python src/train.py --name \${config_name} --hpc
EOF
git add --all
git commit -m "Experiment Tune Config $NAME $ADD_NAME"
echo "#$(git rev-parse HEAD)" >> $PBS_PATH
cd $EXPERIMENT_DIR
echo ${PBS_PATH}
jobid=$(qsub ${PBS_PATH})
echo $jobid

