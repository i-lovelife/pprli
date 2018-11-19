#!/bin/bash
#PBS -P NLH                   # your project name
#PBS -q alloc-dt
#PBS -l select=1:ncpus=8:ngpus=1:mpiprocs=8:mem=45gb  # select one chunk with 1 core and 4 GB memory
#PBS -l walltime=10:00:00         # actual time your job will run for
 
cd "/project/RDS-FEI-NLH-RW/work/pprli/"
module load cuda/9.1.85 openmpi-gcc/3.0.0-cuda python/3.6.5
source pprli_env/bin/activate
python src/models/aeganf.py --gpu 0 train --debug --no-test
