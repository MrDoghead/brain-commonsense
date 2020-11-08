#!/bin/bash

python3 extract_nlp_features.py \
    --sequence_length 1  \
    --output_dir bert

#for((i=20;i<=40;i+=5));
#do
#echo "seq_len: $i";
#python3 extract_nlp_features.py \
#    --sequence_length $i \
#    --output_dir bert;
#done
