import utils
import torch
import config
import string
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from settings import get_module_logger
logger = get_module_logger(__name__)

def loss_fn(o1, o2, t1, t2):
    l1 = nn.BCEWithLogitsLoss()(o1, t1)
    l2 = nn.BCEWithLogitsLoss()(o2, t2)
    return l1 + l2


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    losses = utils.AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    for bi, d in enumerate(tk0):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.float)
        targets_end = targets_end.to(device, dtype=torch.float)

        optimizer.zero_grad()
        o1, o2 = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        loss = loss_fn(o1, o2, targets_start, targets_end)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss = losses.avg)

def eval_fn(data_loader, model, device):
    model.eval()
    fin_outputs_start = []
    fin_outputs_end = []
    fin_tweet_tokens = []
    fin_padding_lens = []
    fin_orig_selected = []
    fin_orig_sentiment = []
    fin_orig_tweet = []
    fin_tweet_token_ids = []
    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        tweet_tokens = d["tweet_tokens"]
        padding_len = d["padding_len"]
        sentiment = d["sentiment"]
        orig_selected = d["orig_selected"]
        orig_sentiment = d["orig_sentiment"]
        orig_tweet = d["orig_tweet"]
       

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
       

        o1, o2 = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        fin_outputs_start.append(torch.sigmoid(o1).cpu().detach().numpy())
        fin_outputs_end.append(torch.sigmoid(o2).cpu().detach().numpy())
        fin_padding_lens.extend(padding_len.cpu().detach().numpy().tolist())
        fin_tweet_token_ids.append(ids.cpu().detach().numpy().tolist())

        fin_tweet_tokens.extend(tweet_tokens)
        fin_orig_sentiment.extend(orig_sentiment)
        fin_orig_selected.extend(orig_selected)
        fin_orig_tweet.extend(orig_tweet)

    fin_outputs_start = np.vstack(fin_outputs_start) # [array([[1, 2, 3],[4, 5, 6]])] --- >  [[1 2 3][4 5 6]] np array
    fin_outputs_end = np.vstack(fin_outputs_end)
    fin_tweet_token_ids = np.vstack(fin_tweet_token_ids)
    jaccards = []
    threshold = 0.2

    for j in range(len(fin_tweet_tokens)):
        logger.info("tweet tokens = {}".format(fin_tweet_tokens[j]))
        target_string = fin_orig_selected[j]
        tweet_tokens = fin_tweet_tokens[j]
        padding_len = fin_padding_lens[j]
        original_tweet = fin_orig_tweet[j]
        sentiment_val = fin_orig_sentiment[j]

        if padding_len > 0:
            mask_start = fin_outputs_start[j, 3:-1][:-padding_len] >= threshold
            mask_end = fin_outputs_end[j, 3:-1][:-padding_len] >= threshold
            tweet_token_ids = fin_tweet_token_ids[j, 3:-1][:-padding_len]
        else:
            mask_start = fin_outputs_start[j, 3:-1] >= threshold
            mask_end = fin_outputs_end[j, 3:-1] >= threshold
            tweet_token_ids = fin_tweet_token_ids[j, 3:-1]

        
        mask = [0] * len(mask_start)

        idx_start = np.nonzero(mask_start)[0]
        idx_end = np.nonzero(mask_end)[0]
        if len(idx_start) > 0:
            idx_start = idx_start[0]
            if len(idx_end) > 0:
                idx_end = idx_end[0]
            else:
                idx_end = idx_start
        else:
            idx_start = 0
            idx_end = 0
        for mj in range(idx_start, idx_end + 1):
            mask[mj] = 1

        output_tokens = [x for p, x in enumerate(tweet_token_ids) if mask[p] == 1]

        filtered_output = config.TOKENIZER.decode(output_tokens)
        filtered_output = filtered_output.strip().lower()
        logger.info("target_string = {} - filtered_output = {}".format(target_string.strip(), filtered_output.strip()))
        if sentiment == "neutral":
            filtered_output = original_tweet
        jac = utils.jaccard(target_string.strip(), filtered_output.strip())
        jaccards.append(jac)
    mean_jac = np.mean(jaccards)
    return mean_jac
        