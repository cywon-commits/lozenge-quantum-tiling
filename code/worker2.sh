i=0
while IFS= read -r line; do
  if [ $((i % $3)) -eq $2 ]; then eval "$line"; fi
  i=$((i+1))
done < $1
