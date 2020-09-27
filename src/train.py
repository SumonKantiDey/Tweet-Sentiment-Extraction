import utils
import config
import dataset
import engine
import torch
import transformers
import pandas as pd
import torch.nn as nn
import numpy as np
from settings import get_module_logger
from model import TweetModel
from sklearn import model_selection
from transformers import AdamW
from dataset import TweetDataset
from transformers import get_linear_schedule_with_warmup

logger = get_module_logger(__name__)


def run(fold):
    dfx = pd.read_csv(config.TRAINING_FILE)

    # Set train validation set split
    df_train = dfx[dfx.kfold != fold].reset_index(drop=True)
    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)

    train_dataset = TweetDataset(
        tweet=df_train.text.values,
        sentiment=df_train.sentiment.values,
        selected_text=df_train.selected_text.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4
    )

    valid_dataset = TweetDataset(
        tweet=df_valid.text.values,
        sentiment=df_valid.sentiment.values,
        selected_text=df_valid.selected_text.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=2
    )

    device = torch.device("cuda")
    model_config = transformers.BertConfig.from_pretrained(config.ROBERTA_PATH)
    model_config.output_hidden_states = True
    model = TweetModel(conf=model_config)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    # Define two sets of parameters: those with weight decay, and those without
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(
        len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    '''
    Create a scheduler to set the learning rate at each training step
    "Create a schedule with a learning rate that decreases linearly after linearly increasing during a warmup period." (https://pytorch.org/docs/stable/optim.html)
    Since num_warmup_steps = 0, the learning rate starts at 3e-5, and then linearly decreases at each training step
    '''
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )
    es = utils.EarlyStopping(patience=2, mode="max")
    print(f"Training is Starting for fold={fold}")
    logger.info("{} - {}".format("Training is Starting for fold", fold))
    #model=nn.DataParallel(model)

    for epoch in range(3):
        engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        jaccard=engine.eval_fn(valid_data_loader, model, device)
        print(f"Jaccard Score = {jaccard}")
        logger.info("EPOCHS {} - Jaccard Score - {}".format(epoch, jaccard))
        es(jaccard, model, model_path=f"../models/nmodel_{fold}.bin")
        if es.early_stop:
            print("Early stopping")
            break

if __name__ == "__main__":
    for fold in range(2,5):
        run(fold)
