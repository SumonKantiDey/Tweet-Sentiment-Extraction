import utils
import torch
import config
import string
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from settings import get_module_logger
logger = get_module_logger(__name__)

def loss_fn(start_logits, end_logits, start_positions, end_positions):
    loss_fct = nn.CrossEntropyLoss()
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss)
    return total_loss


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()

    losses = utils.AverageMeter()
    jaccards = utils.AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader))

    for bi, d in enumerate(tk0):
        ids = d["ids"]
        mask = d["mask"]
        token_type_ids = d["token_type_ids"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]
        orig_tweet = d["orig_tweet"]
        orig_selected = d["orig_selected"]
        sentiment = d["sentiment"]
        offsets = d["offsets"].numpy()
        

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)
        # Reset gradients
        model.zero_grad()
        # Use ids, masks, and token types as input to the model
        # Predict logits for each of the input tokens for each batch
        outputs_start, outputs_end = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids,
        )

        loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end) # Calculate batch loss based on CrossEntropy
        loss.backward() # Calculate gradients based on loss
        optimizer.step() # Adjust weights based on calculated gradients
        scheduler.step() # Update scheduler

        '''
        Apply softmax to the start and end logits
        This squeezes each of the logits in a sequence to a value between 0 and 1, while ensuring that they sum to 1
        This is similar to the characteristics of "probabilities"
        '''
        outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
        outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
        jaccard_scores = [] # Calculate the jaccard score based on the predictions for this batch
        
        for px, tweet in enumerate(orig_tweet):
            selected_tweet = orig_selected[px]
            tweet_sentiment = sentiment[px]
            jaccard_score, _ = utils.calculate_jaccard_score(
                original_tweet=tweet,
                target_string=selected_tweet,
                sentiment_val=tweet_sentiment,
                idx_start=np.argmax(outputs_start[px, :]),
                idx_end=np.argmax(outputs_end[px, :]),
                offsets=offsets[px]
            )
            jaccard_scores.append(jaccard_score)
        jaccards.update(np.mean(jaccard_scores), ids.size(0))
        losses.update(loss.item(), ids.size(0))  # ids.size(0) means current batch size
        tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg) # Print the average loss and jaccard score at the end of each batch

def eval_fn(data_loader, model, device):
    model.eval()

    losses = utils.AverageMeter()
    jaccards = utils.AverageMeter()
    with torch.no_grad(): #https://datascience.stackexchange.com/questions/32651/what-is-the-use-of-torch-no-grad-in-pytorch
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            ids = d["ids"]
            mask = d["mask"]
            token_type_ids = d["token_type_ids"]
            targets_start = d["targets_start"]
            targets_end = d["targets_end"]
            orig_tweet = d["orig_tweet"]
            orig_selected = d["orig_selected"]
            sentiment = d["sentiment"]
            offsets = d["offsets"].numpy()
            

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets_start = targets_start.to(device, dtype=torch.long)
            targets_end = targets_end.to(device, dtype=torch.long)
        
            outputs_start, outputs_end = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end) # Calculate batch loss based on CrossEntropy
            '''
            Apply softmax to the start and end logits
            This squeezes each of the logits in a sequence to a value between 0 and 1, while ensuring that they sum to 1
            This is similar to the characteristics of "probabilities"
            '''
            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
            
            jaccard_scores = [] # Calculate the jaccard score based on the predictions for this batch
            for px, tweet in enumerate(orig_tweet):
                selected_tweet = orig_selected[px]
                tweet_sentiment = sentiment[px]
                jaccard_score, _ = utils.calculate_jaccard_score(
                    original_tweet=tweet,
                    target_string=selected_tweet,
                    sentiment_val=tweet_sentiment,
                    idx_start=np.argmax(outputs_start[px, :]),
                    idx_end=np.argmax(outputs_end[px, :]),
                    offsets=offsets[px]
                )
                jaccard_scores.append(jaccard_score)
            jaccards.update(np.mean(jaccard_scores), ids.size(0))
            losses.update(loss.item(), ids.size(0))  # ids.size(0) means current batch size
            tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg) # Print the average loss and jaccard score at the end of each batch
    print(f"Jaccard = {jaccards.avg}")
    logger.info("{} = {}".format("Jaccard", jaccards.avg))
    return jaccards.avg