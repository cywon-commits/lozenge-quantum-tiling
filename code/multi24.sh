while pgrep -f "rest18.sh" >/dev/null; do sleep 15; done
python tilt.py 24 '[[0,6],[12,6]]' 24 0.10 1500 81 >> multi24.log 2>&1
python tilt.py 24 '[[0,6],[9,6]]' 24 0.10 1500 82 >> multi24.log 2>&1
python tilt.py 24 '[[0,12]]' 24 0.10 1500 83 >> multi24.log 2>&1
python tilt.py 24 '[[0,6],[12,6]]' 24 0.20 1500 84 >> multi24.log 2>&1
python tilt.py 24 '[[0,6],[9,6]]' 24 0.20 1500 85 >> multi24.log 2>&1
