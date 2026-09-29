#!/bin/bash
cd "$(dirname "$0")/../data"
until grep -q '^{' qbig_m36.log 2>/dev/null; do python3 qmc_ckpt.py big_m36_L36.pkl 36 36 0.13 1200 6000 1 >> qbig_m36.log 2>&1; sleep 2; done
