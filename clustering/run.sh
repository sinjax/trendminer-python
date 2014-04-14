echo 'Step 0: Processing data - tokenization, language detection and filtering'
echo 'If this is not done, this is done the Trendminer Preprocessing Pipeline'
echo 'Step 1: Deduplication of tweets'
cat $1 | python dedup.py > $1-dedup
echo 'Step 2: Computing word counts'
cat $1-dedup | python wc-map.sh | sort | python wc-red.sh > wc.$1-dedup
echo 'Step 3: Creating dictionary'
cat wc.$1-dedup | python wc-filter.py $2 | python makevoc.py > dict.$1
echo 'Step 4: Computing dictionary unigram counts'
cat $1-dedup | python wc-map-dict.sh dict.$1 | sort | python wc-red.sh > wc.$1-dedup
echo 'Step 5: Computing dictionary co-occurence counts'
cat $1-dedup | python pmi-map-ml.sh dict.$1 | sort | python wc-red.sh > wco.$1-dedup
echo 'Step 6: Computing NPMI scores'
python compute-npmi.py wc.$1-dedup wco.$1-dedup > npmi.$1-dedup
echo 'Step 7: Truncating and making NPMI symmetric'
cat npmi.$1-dedup | python truncate.py $3 | python sym.py > npmi.$1-dedup-sym-$3
echo 'Step 8: Performing the actual clustering'
#matlab -nojvm -nodesktop -nosplash -nodisplay -r "spectral('$1','npmi.$1-dedup-sym-$3','dict.$1',$4,$5); quit;"
echo python run_spectral.py -tweets $1 -npmi npmi.$1-dedup-sym-$3 -dict dict.$1 -k $4 -c $5
echo "Finished, clustering results are in cl.$1-$5"
