#!/bin/bash
cd "$(dirname "$0")/../data"
until grep -q ALLDONE per_cell.log 2>/dev/null; do python3 per_cell.py >> per_cell.log 2>&1; sleep 2; done
