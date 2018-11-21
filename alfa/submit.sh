#!/bin/bash
git add --all
git commit -m 'experiment'
git push
ssh lnan6257@hpc.sydney.edu.au "cd /project/RDS-FEI-NLH-RW/work/pprli; git pull; cd hpc_scripts/; qsub $1 2>&1 "