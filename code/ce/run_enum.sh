#!/bin/bash
# Class-resolved enumeration of all supercells with <= 36 sites; run from data/ce/
C="$(cd "$(dirname "$0")/.." && pwd)"
python -c "
import sys; sys.path.insert(0,'$C')
from periodic import cells
for p,q,s in cells(36,36): print(p,q,s)" > cells36.txt
wc -l cells36.txt
cat cells36.txt | xargs -P 2 -L 1 bash -c "python $C/enum_nb.py \$0 \$1 \$2 --keep 0.03 > /dev/null 2>&1"
