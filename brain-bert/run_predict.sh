#!/bin/bash

for((i=1;i<=12;i++));
do
python3 predict_brain_from_nlp.py \
    --subject M \
    --nlp_feat_type bert \
    --nlp_feat_dir bert \
    --layer $i \
    --sequence_length 10 \
    --output_dir output_bert/;
done
