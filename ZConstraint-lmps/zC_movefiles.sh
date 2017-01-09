#PBS -l nodes=1,walltime=0:01:00
#PBS -N movingfiles
#PBS -q batch
#PBS -j oe
#PBS -A BIP140
#PBS -V

cd $PBS_O_WORKDIR
cd Sim1
mv forceout* ../Forces/
mv restartfile ../Forces/restartfile1
mv outputfile ../Forces/outputfile1
mv trajectory.lammps ../Forces/trajectory_1.lammps
cd ..
cd Sim2
mv forceout* ../Forces/
mv restartfile ../Forces/restartfile2
mv outputfile ../Forces/outputfile2
mv trajectory.lammps ../Forces/trajectory_2.lammps
cd ..
cd Sim3
mv forceout* ../Forces/
mv restartfile ../Forces/restartfile3
mv outputfile ../Forces/outputfile3
mv trajectory.lammps ../Forces/trajectory_3.lammps
cd ..
cd Sim4
mv forceout* ../Forces/
mv restartfile ../Forces/restartfile4
mv outputfile ../Forces/outputfile4
mv trajectory.lammps ../Forces/trajectory_4.lammps
cd ..
cd Sim5
mv forceout* ../Forces/
mv restartfile ../Forces/restartfile5
mv outputfile ../Forces/outputfile5
mv trajectory.lammps ../Forces/trajectory_5.lammps
cd ..
