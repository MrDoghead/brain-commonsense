## Session2

In this session, we are going to fine-tune bert on three different commonsense tasks, which are object-property, object-affordance, property-affordance. Then compare the abilities of different layers when learning commonsense.

## To run the prediction

Please specify the data, task and the layer before prediction.

### Run the baselines: random and majority.
python -m pc.baselines

### Display human results.
python -m pc.human

### Run BERT. NOTE: 1 epoch for "situated-AP" is not to handicap the model; rather, it overfits and achieves 0.0 F1 score for epoch 2+.
python -m pc.bert --task "abstract-OP" --epochs 5
python -m pc.bert --task "situated-OP" --epochs 5
python -m pc.bert --task "situated-OA" --epochs 5
python -m pc.bert --task "situated-AP" --epochs 1
