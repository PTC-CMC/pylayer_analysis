#SBATCH -p regular 
#SBATCH -N 1 
#SBATCH -t 08:00:00 
#SBATCH -J Stage5_ZCon0 
#SBATCH -o Stage5_ZCon0 
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=alexander.h.yang@vanderbilt.edu

module load gromacs/5.1.2
srun -n 12 mdrun_mpi_sp -ntmpi 12 -cpi Stage4_Eq0.cpt -deffnm Stage5_ZCon0
