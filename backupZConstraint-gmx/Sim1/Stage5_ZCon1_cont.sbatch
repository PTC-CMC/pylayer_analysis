#SBATCH -p regular 
#SBATCH -N 1 
#SBATCH -t 08:00:00 
#SBATCH -J Stage5_ZCon1 
#SBATCH -o Stage5_ZCon1 
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=alexander.h.yang@vanderbilt.edu

module load gromacs/5.1.2
srun -n 12 mdrun_mpi_sp -ntmpi 12 -append -s Stage5_ZCon1.tpr -cpi Stage5_ZCon1.cpt
