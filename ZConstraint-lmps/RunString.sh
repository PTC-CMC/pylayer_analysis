#!/bin/bash
FIRST=`qsub zC_Eq_sims.sh`
echo $FIRST
SECOND=`qsub -W depend=afterok:$FIRST zC_MD_sims.sh`
echo $SECOND
THIRD=`qsub -W depend=afterok:$SECOND zC_MD_sims.sh`
echo $THIRD
FOURTH=`qsub -W depend=afterok:$THIRD zC_movefiles.sh`
echo $FOURTH
exit 0
