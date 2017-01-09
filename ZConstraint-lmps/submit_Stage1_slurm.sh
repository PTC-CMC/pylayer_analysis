#!bin/bash -l
#SBATCH -p regular
#SBATCH -N 2
#SBATCH -t 06:00:00
#SBATCH -J zConstStage1
#SBATCH -o zConstStage1.o%j

module load lammps/20160514

python Stage1_setTracers.py
