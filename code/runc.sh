#!/bin/bash
cd /mnt/user-data/outputs/.qdec
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NLDIV=50
cj=$1; k=$2; beta=$3; h=$4; sd=${5:-1}
tag=$(basename $cj .json)_k${k}_b${beta}_h${h}_s${sd}
until grep -q '^{' c_${tag}.log 2>/dev/null; do python3 /home/claude/k2/qdec_cell.py $cj $k $beta $h ${NT:-600} ${NM:-3000} $sd >> c_${tag}.log 2>&1; sleep 2; done
