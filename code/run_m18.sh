#!/bin/bash
cd "$(dirname "$0")/../data"
until grep -q '^{' qbig_m18.log 2>/dev/null; do python3 qmc_ckpt.py big_m18_L36.pkl 36 36 0.14 1200 6000 1 >> qbig_m18.log 2>&1; sleep 2; done
