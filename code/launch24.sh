while pgrep -f "worker.sh jobs.txt 0" >/dev/null; do sleep 20; done
sh worker.sh jobs24.txt 0 1 field24.log
