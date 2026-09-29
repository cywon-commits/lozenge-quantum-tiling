#!/bin/bash
cd "$(dirname "$0")/../data"
until python3 per_hist.py >> per_hist.log 2>&1; do sleep 2; done
echo ALLDONE >> per_hist.log
