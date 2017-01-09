#!/bin/sh -l

echo 'hi' > Sim0/Stage1_Weak0.cpt
echo 'hi' > Sim0/Stage1_Weak0.tpr
echo 'hi' > Sim1/Stage1_Weak1.cpt
echo 'hi' > Sim1/Stage1_Weak1.tpr
echo 'hi' > Sim2/Stage1_Weak2.cpt
echo 'hi' > Sim2/Stage1_Weak2.tpr
echo 'hi' > Sim3/Stage1_Weak3.cpt
echo 'hi' > Sim3/Stage1_Weak3.tpr
echo 'hi' > Sim4/Stage1_Weak4.cpt
echo 'hi' > Sim4/Stage1_Weak4.tpr

cp pureDSPC.gro Sim0/Stage1_Weak0.gro
cp pureDSPC.gro Sim1/Stage1_Weak1.gro
cp pureDSPC.gro Sim2/Stage1_Weak2.gro
cp pureDSPC.gro Sim3/Stage1_Weak3.gro
cp pureDSPC.gro Sim4/Stage1_Weak4.gro
