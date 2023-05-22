for DIR in ./dumped/ode2_data/*digits
do
cat ${DIR}/*/data.prefix \
| awk 'BEGIN{PROCINFO["sorted_in"]="@val_num_desc"}{c[$0]++}END{for (i in c) printf("%i|%s\n",c[i],i)}' \
> ${DIR}/data.prefix.counts

# create a valid and a test set of 10k equations
python split_data.py ${DIR}/data.prefix.counts $(awk 'END {print int(NR*0.2)}' ${DIR}/data.prefix.counts)

# remove valid inputs that are in the train
mv ${DIR}/data.prefix.counts.valid ${DIR}/data.prefix.counts.valid.old
awk -F"[|\t]" 'NR==FNR { lines[$2]=1; next } !($2 in lines)' <(cat ${DIR}/data.prefix.counts.train) ${DIR}/data.prefix.counts.valid.old \
> ${DIR}/data.prefix.counts.valid

# remove test inputs that are in the train
mv ${DIR}/data.prefix.counts.test ${DIR}/data.prefix.counts.test.old
awk -F"[|\t]" 'NR==FNR { lines[$2]=1; next } !($2 in lines)' <(cat ${DIR}/data.prefix.counts.train) ${DIR}/data.prefix.counts.test.old \
> ${DIR}/data.prefix.counts.test
done