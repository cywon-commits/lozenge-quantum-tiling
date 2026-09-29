# usage: sh worker.sh jobfile workerid nworkers logfile
i=0
while IFS= read -r line; do
  if [ $((i % $3)) -eq $2 ]; then eval "$line" >> $4 2>&1; fi
  i=$((i+1))
done < $1
