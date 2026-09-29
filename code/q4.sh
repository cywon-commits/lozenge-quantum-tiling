while pgrep -f "sh q2.sh" >/dev/null; do sleep 20; done
python qmc_tiling.py 18 dice 18 0 2000
python tilt.py 18 '[[0,12]]' 18 0 2000
python qmc_tiling.py 18 stripe 18 0 2000
