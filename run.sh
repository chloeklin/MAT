#!/bin/bash

#PBS -q gpuvolta
#PBS -P um09
#PBS -l ncpus=48
#PBS -l ngpus=4
#PBS -l mem=128GB
#PBS -l walltime=48:00:00
#PBS -l storage=scratch/um09
#PBS -l jobfs=100GB

cd /scratch/um09/hl4138

module load python3/3.7.4 cuda/10.1
module list

source mat-venv/bin/activate
cd MAT/src/

# Run the Python script with the current parameters
python3 finetune.py --target u0 --model "Molecule Attention Transformer"