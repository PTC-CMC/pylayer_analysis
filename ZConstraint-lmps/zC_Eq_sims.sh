#PBS -l nodes=5,walltime=1:30:00
#PBS -N sims_Eq
#PBS -q batch
#PBS -j oe
#PBS -A BIP140
#PBS -V

cd $PBS_O_WORKDIR
export CRAY_CUDA_MPS=1
cd $PBS_O_WORKDIR
cd Sim1
aprun -n 16 -N 16 lmp_titan -sf gpu -pk gpu 1 -in inStage3_equi.dat -var ID1 1454 -var force1 forceout0 -var zt1 -34.00 -var ID2 2248 -var force2 forceout5 -var zt2 -24.00 -var ID3 2236 -var force3 forceout10 -var zt3 -14.00 -var ID4 1794 -var force4 forceout15 -var zt4 -4.00 -var ID5 2518 -var force5 forceout20 -var zt5 6.00 -var ID6 1505 -var force6 forceout25 -var zt6 16.00 -var ID7 2534 -var force7 forceout30 -var zt7 26.00  -var restartfile restartfile &
cd ..
cd $PBS_O_WORKDIR
cd Sim2
aprun -n 16 -N 16 lmp_titan -sf gpu -pk gpu 1 -in inStage3_equi.dat -var ID1 1454 -var force1 forceout1 -var zt1 -32.00 -var ID2 2248 -var force2 forceout6 -var zt2 -22.00 -var ID3 2236 -var force3 forceout11 -var zt3 -12.00 -var ID4 1794 -var force4 forceout16 -var zt4 -2.00 -var ID5 2518 -var force5 forceout21 -var zt5 8.00 -var ID6 1505 -var force6 forceout26 -var zt6 18.00 -var ID7 2534 -var force7 forceout31 -var zt7 28.00  -var restartfile restartfile &
cd ..
cd $PBS_O_WORKDIR
cd Sim3
aprun -n 16 -N 16 lmp_titan -sf gpu -pk gpu 1 -in inStage3_equi.dat -var ID1 1454 -var force1 forceout2 -var zt1 -30.00 -var ID2 2248 -var force2 forceout7 -var zt2 -20.00 -var ID3 2236 -var force3 forceout12 -var zt3 -10.00 -var ID4 1794 -var force4 forceout17 -var zt4 0.00 -var ID5 2518 -var force5 forceout22 -var zt5 10.00 -var ID6 1505 -var force6 forceout27 -var zt6 20.00 -var ID7 2534 -var force7 forceout32 -var zt7 30.00  -var restartfile restartfile &
cd ..
cd $PBS_O_WORKDIR
cd Sim4
aprun -n 16 -N 16 lmp_titan -sf gpu -pk gpu 1 -in inStage3_equi.dat -var ID1 1454 -var force1 forceout3 -var zt1 -28.00 -var ID2 2248 -var force2 forceout8 -var zt2 -18.00 -var ID3 2236 -var force3 forceout13 -var zt3 -8.00 -var ID4 1794 -var force4 forceout18 -var zt4 2.00 -var ID5 2518 -var force5 forceout23 -var zt5 12.00 -var ID6 1505 -var force6 forceout28 -var zt6 22.00 -var ID7 2534 -var force7 forceout33 -var zt7 32.00  -var restartfile restartfile &
cd ..
cd $PBS_O_WORKDIR
cd Sim5
aprun -n 16 -N 16 lmp_titan -sf gpu -pk gpu 1 -in inStage3_equi.dat -var ID1 1454 -var force1 forceout4 -var zt1 -26.00 -var ID2 2248 -var force2 forceout9 -var zt2 -16.00 -var ID3 2236 -var force3 forceout14 -var zt3 -6.00 -var ID4 1794 -var force4 forceout19 -var zt4 4.00 -var ID5 2518 -var force5 forceout24 -var zt5 14.00 -var ID6 1505 -var force6 forceout29 -var zt6 24.00 -var ID7 2534 -var force7 forceout34 -var zt7 34.00  -var restartfile restartfile &
cd ..
wait
