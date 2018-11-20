#!/bin/bash
#PBS -P NLH
#PBS -q alloc-dt
#PBS -l select=1:ncpus=8:ngpus=1:mpiprocs=8:mem=45gb
#PBS -l walltime=10:00:00
 
cd "/project/RDS-FEI-NLH-RW/work/pprli/"
module load cuda/9.1.85 openmpi-gcc/3.0.0-cuda python/3.6.5
source pprli_env/bin/activate
python src/models/aeganf.py --gpu 0 train --debug --no-test --epoch 101
