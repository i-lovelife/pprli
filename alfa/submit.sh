#!/bin/bash
git add --all
git commit -m 'experiment'
git push
scp $1 lnan6257@hpc.sydney.edu.au:/