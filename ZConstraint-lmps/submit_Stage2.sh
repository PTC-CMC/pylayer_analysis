#PBS -N zConstStage2
#PBS -q batch
#PBS -l nodes=1,walltime=01:00:00
#PBS -j oe
#PBS -A BIP140
# #PBS -m ae
#PBS -V

cd $PBS_O_WORKDIR
module swap PrgEnv-pgi PrgEnv-gnu
module unload python/3.5.1
module load python_numpy/1.9.2
module load fftw
module load lammps
#module load python_anaconda3/2.3.0

export CRAY_CUDA_MPS=1

python Stage2_shiftTracers.py
