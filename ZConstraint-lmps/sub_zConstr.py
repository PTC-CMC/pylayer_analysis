from __future__ import print_function
from os import system

def printHeader(f, nodes=5, lammpsgpu=True, 
        hrs=2, mins=00, secs=0, job_name='window',
        mail='ae', queue='batch', mail_address='remco.hartkamp@vanderbilt.edu',
        output_file='run.log', env=True, input_file=None,
        extra_commands=None, after_commands=None, lammpscommand='empty'):

    print('#PBS -l nodes=%d,walltime=%d:%02d:%02d' % (nodes, hrs, mins, secs), file=f)
    print('#PBS -N %s' % job_name, file=f)
    print('#PBS -q %s' % queue, file=f)
    print('#PBS -j oe', file=f)
    print('#PBS -A BIP140', file=f)
    print('#PBS -V', file=f)
    print('', file=f)
    print('cd $PBS_O_WORKDIR', file=f)
    if lammpsgpu:
        #print('module swap PrgEnv-pgi PrgEnv-gnu', file=f)
        #print('module load fftw', file=f)
        #print('module load lammps', file=f)
        print('export CRAY_CUDA_MPS=1', file=f)

def submit_Equi(filename='submit.sh', cores=16,
        hrs=2, mins=00, secs=0, job_name='window',
        mail='ae', queue='batch', mail_address='remco.hartkamp@vanderbilt.edu',
        output_file='run.log', env=True, input_file=None,
        extra_commands=None, after_commands=None, lammpscommand='empty'):

    f = open(filename, 'w')
    print('#PBS -l nodes=1,walltime=%d:%02d:%02d' % (hrs, mins, secs), file=f)
    print('#PBS -N %s' % job_name, file=f)
    print('#PBS -q %s' % queue, file=f)
    print('#PBS -j oe', file=f)
    print('#PBS -A BIP140', file=f)
    print('#PBS -V', file=f)
    print('', file=f)
    print('cd $PBS_O_WORKDIR', file=f)
    print('module swap PrgEnv-pgi PrgEnv-gnu', file=f)
    print('module load fftw', file=f)
    print('module load lammps', file=f)
    print('export CRAY_CUDA_MPS=1', file=f)
    print('%s' % lammpscommand, file=f)
    #print('qsub submit2.sh', file=f)

    f.close()

def submit_MDrun(filename='submit2.sh', cores=16,
        hrs=2, mins=00, secs=0, job_name='window',
        mail='ae', queue='batch', mail_address='remco.hartkamp@vanderbilt.edu',
        output_file='run.log', env=True, input_file=None,
        extra_commands=None, after_commands=None, lammpscommand='empty'):

    f = open(filename, 'w')
    print('#PBS -l nodes=1,walltime=%d:%02d:%02d' % (hrs, mins, secs), file=f)
    print('#PBS -N %s' % job_name, file=f)
    print('#PBS -q %s' % queue, file=f)
    print('#PBS -j oe', file=f)
    print('#PBS -A BIP140', file=f)
    print('#PBS -V', file=f)
    print('', file=f)
    print('cd $PBS_O_WORKDIR', file=f)
    print('module swap PrgEnv-pgi PrgEnv-gnu', file=f)
    print('module load fftw', file=f)
    print('module load lammps', file=f)
    print('export CRAY_CUDA_MPS=1', file=f)
    print('%s' % lammpscommand, file=f)

    if after_commands:
        for command in after_commands:
            print(command, file=f)

    f.close()
    system('./RunString.sh') 
