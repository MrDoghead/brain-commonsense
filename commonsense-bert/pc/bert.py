import argparse
import code  
import csv
from datetime import datetime
import os,sys
import random
import time
import logging
from typing import List, Tuple, Dict, Set, Any, Optional, Callable

import numpy as np
from pytorch_transformers import BertTokenizer, BertForSequenceClassification
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from tensorboardX import SummaryWriter
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn import model_selection

from pc.data import (
    Task,
    get,
    TASK_SHORTHAND,
    TASK_MEDIUMHAND,
    TASK_REV_MEDIUMHAND,
    TASK_LABELS,
)
from pc import metrics
from pc import util

def mlp(
    d_in: int,
    input_dropout: float,
    h: int,
    activation: Any,
    inner_dropout: float,
    d_out: int,
) -> nn.Module:
    return nn.Sequential(
        nn.Dropout(input_dropout),
        nn.Linear(d_in, h),
        activation(),
        nn.Dropout(inner_dropout),
        nn.Linear(h, d_out),
        nn.Sigmoid(),
    )

class BertDataset(Dataset):
    def __init__(self, task: Task, train: bool, seq_len: int = 20) -> None:
        self.seq_len = seq_len

        # load labels and y data
        train_data, test_data = get(task)
        split_data = train_data if train else test_data
        self.labels, self.y = split_data
        assert len(self.labels) == len(self.y)

        # load X index
        # line_mapping maps from word1/word2 label to sentence index in sentence list.
        self.line_mapping: Dict[str, int] = {}
        task_short = TASK_SHORTHAND[task]
        with open("data/sentences/index.csv", "r") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if row["task"] == task_short:
                    self.line_mapping[row["uids"]] = i
                    # TODO: check that i lines up and isn't off by one

        with open("data/sentences/sentences.txt", "r") as f:
            self.sentences = [line.strip() for line in f.readlines()]

        '''
        n_sample = 5
        print("{} Samples:".format(n_sample))
        for i in random.sample(range(len(self.labels)), n_sample):
            label = self.labels[i]
            sentence = self.sentences[self.line_mapping[label]]
            print('- {}: "{}"'.format(label, sentence))
        '''

        print("Loading tokenizer...")
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-large-uncased", do_lower_case=True, do_basic_tokenize=True
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        label = self.labels[i]

        # tokenize
        max_sent_len = self.seq_len - 2
        sentence = self.sentences[self.line_mapping[label]]
        tkns = ["[CLS]"] + self.tokenizer.tokenize(sentence)[:max_sent_len] + ["[SEP]"]

        input_mask = [1] * len(tkns)

        # padding
        if len(tkns) < self.seq_len:
            pad_len = self.seq_len - len(tkns)
            tkns += ["[PAD]"] * pad_len
            input_mask += [0] * pad_len

        # code.interact(local=dict(globals(), **locals()))
        input_ids = self.tokenizer.convert_tokens_to_ids(tkns)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "input_mask": torch.tensor(input_mask, dtype=torch.long),
            "label": label,
            "y": self.y[i],
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=TASK_REV_MEDIUMHAND.keys(),
        help="Name of task to run",
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=5, help="How many epochs to run")
    parser.add_argument("--layer", type=int, default=12, help="Which bert layer to run")
    args = parser.parse_args()
    task = TASK_REV_MEDIUMHAND[args.task]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initial_lr = 5e-5
    warmup_proportion = 0.1
    train_batch_size = 64
    test_batch_size = 96
    train_epochs = args.epochs

    print("Building model...")
    model = mlp(600, 0.0, 128, nn.ReLU, 0.0, 1)
    logging.info("Model:")
    logging.info(model)
    model.to(device)

    print("Loading traning data")
    train_dataset = BertDataset(task, True)
    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=8
    )
    print("Loading test data")
    test_dataset = BertDataset(task, False)
    test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=8
    )

    labels: List[str] = []
    # training
    for batch_i, batch in enumerate(tqdm(train_loader, desc="Batch")):
        input_ids = batch["input_ids"].to(device)
        y = batch["y"].to(device, dtype=torch.long)

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.adam(lr = 0.0001)

        x = torch.from_numpy(input_ids).float().to(device)
        y = torch.from_numpy(y).float().to(device)
        assert x.shape[0] == y.shape[0]

        # training
        model.train()
        batch_y_hat = model(batch_x)
        loss = loss_fn(batch_y_hat, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # testing
    for batch_i, batch in enumerate(tqdm(test_loader, desc="Batch")):
        input_ids = batch["input_ids"].to(device)
        y = batch["y"].to(device, dtype=torch.long)
        x = torch.from_numpy(input_ids).float().to(DEVICE)
        y = torch.from_numpy(y).float().to(device)

        model.eval()
        y_hat = model(x).round().int().cpu().numpy()

        metrics.report(y_test_hat, y_test, labels_test, data.TASK_LABELS[task])
    
    #metrics.report(overall_y_hat, y_test, labels_test, data.TASK_LABELS[task])


if __name__ == "__main__":
    main()
