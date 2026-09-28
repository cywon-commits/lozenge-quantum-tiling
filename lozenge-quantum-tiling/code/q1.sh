for f in 0.125 0.25 0.375 0.5; do python qmc_tiling.py 12 random 12 0 3000 $f 1; done
python qmc_tiling.py 12 dice 24 0 3000
for h in 0.1 0.2 0.3 0.4; do python qmc_tiling.py 12 stripe 12 $h 3000; done
