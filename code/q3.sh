while pgrep -f "sh q1.sh" >/dev/null; do sleep 20; done
python tilt.py 12 '[[0,6]]' 12 0 3000
python qmc_tiling.py 12 square 12 0 3000
python tilt.py 18 '[[0,6]]' 18 0 2000
python tilt.py 18 '[[0,18]]' 18 0 2000
