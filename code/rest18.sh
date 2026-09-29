while kill -0 579 2>/dev/null; do sleep 15; done
for h in 0.20 0.23 0.26; do python tilt.py 18 '[[0,6]]' 18 $h 1500 70 >> field18_w1.log 2>&1; done
