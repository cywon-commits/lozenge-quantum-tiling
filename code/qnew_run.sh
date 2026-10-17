#!/bin/bash
# L=36 QMC of the LSWT-best M/6 and M/3 crystals (data/qnew/). Usage: bash qnew_run.sh M6 0.13 1
cd "$(dirname "$0")/../data/qnew"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
tag=$1; h=$2; sd=$3
until grep -q '^{' q_${tag}_s${sd}.log 2>/dev/null; do python3 ../../code/qmc_ckpt.py lswtbest_${tag}_L36.pkl 36 36 $h 1200 6000 $sd >> q_${tag}_s${sd}.log 2>&1; sleep 2; done
