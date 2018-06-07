#!/bin/bash

#SBATCH --nodes=1
#SBATCH --output="matlab%j.out"
#SBATCH --job-name=matlab_gen
#SBATCH --mem=4G

module load matlab/matlab-R2016a

srun matlab -nodesktop -nosplash -nodisplay -r "run('gettrainingexample.m');exit"
