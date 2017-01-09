module swap PrgEnv-pgi PrgEnv-gnu
module unload python/3.5.1
#module load python_numpy/1.9.2
module load fftw
module load lammps
module load python_anaconda3/2.3.0
python Stage3_runs.py
