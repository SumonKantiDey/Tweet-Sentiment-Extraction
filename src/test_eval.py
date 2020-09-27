import config
import model
import transformers
import torch
import utils
import pandas as pd
from model import TweetModel
from dataset import TweetDataset
from tqdm import tqdm
import numpy as np
df_test = pd.read_csv(config.TEST_FILE)
df_test.loc[:, "selected_text"] = df_test.text.values
#data = [["Its coming out the socket I feel like my phones hole is not a virgin. That`s how loose it is... :`(","loose it is...", "negative"]]
# Create the pandas DataFrame 
# df_test = pd.DataFrame(data, columns = ["text","selected_text","sentiment"]) 
# df_test.loc[:, "selected_text"] = df_test.text.values
device = torch.device("cuda")
model_config = transformers.RobertaConfig.from_pretrained(config.ROBERTA_PATH)
model_config.output_hidden_states = True

# Load each of the five trained models and move to GPU
model1 = TweetModel(conf=model_config)
model1.to(device)
model1.load_state_dict(torch.load("../models/nmodel_0.bin")) #strict=False
# print(model1.eval())


model2 = TweetModel(conf=model_config)
model2.to(device)
model2.load_state_dict(torch.load("../models/nmodel_1.bin")) #strict=False
# print(model2.eval())

model3 = TweetModel(conf=model_config)
model3.to(device)
model3.load_state_dict(torch.load("../models/nmodel_2.bin"))
# print(model3.eval())

model4 = TweetModel(conf=model_config)
model4.to(device)
model4.load_state_dict(torch.load("../models/nmodel_3.bin"))
# print(model4.eval())

model5 = TweetModel(conf=model_config)
model5.to(device)
model5.load_state_dict(torch.load("../models/nmodel_4.bin"))
# print(model5.eval())

final_output = []

# Instantiate TweetDataset with the test data
test_dataset = TweetDataset(
        tweet=df_test.text.values,
        sentiment=df_test.sentiment.values,
        selected_text=df_test.selected_text.values
)

# Instantiate DataLoader with `test_dataset`
data_loader = torch.utils.data.DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=config.VALID_BATCH_SIZE,
    num_workers=1
)

# Turn of gradient calculations
with torch.no_grad():
    tk0 = tqdm(data_loader, total=len(data_loader))
    # Predict the span containing the sentiment for each batch
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
        # print("starts = ", targets_start)
        # print("Ends = ", targets_end)


        # Predict start and end logits for each of the five models
        outputs_start1, outputs_end1 = model1(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start2, outputs_end2 = model2(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start3, outputs_end3 = model3(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start4, outputs_end4 = model4(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start5, outputs_end5 = model5(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        # Get the average start and end logits across the five models and use these as predictions
        # This is a form of "ensembling"
        outputs_start = ( outputs_start1 + outputs_start2 + outputs_start3 + outputs_start4 + outputs_start5) / 5
        outputs_end = (outputs_end1 + outputs_end2 + outputs_end3 + outputs_end4 + outputs_end5) / 5
        
        # Apply softmax to the predicted start and end logits
        outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
        outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
        #print(np.argmax(outputs_start[0, :]),np.argmax(outputs_end[0, :]))
        # Convert the start and end scores to actual predicted spans (in string form)
        for px, tweet in enumerate(orig_tweet):
            selected_tweet = orig_selected[px]
            tweet_sentiment = sentiment[px]
            _, output_sentence = utils.calculate_jaccard_score(
                original_tweet=tweet,
                target_string=selected_tweet,
                sentiment_val=tweet_sentiment,
                idx_start=np.argmax(outputs_start[px, :]),
                idx_end=np.argmax(outputs_end[px, :]),
                offsets=offsets[px],
                verbose=True
            )
            final_output.append(output_sentence)

def post_process(selected):
    return " ".join(set(selected.lower().split()))
#print(final_output)
#df_test.loc[:, 'selected_text'] = final_output
#df_test.selected_text = df_test.selected_text.map(post_process)
sample = pd.read_csv("../input/sample_submission.csv")
sample.loc[:, 'selected_text'] = final_output
#sample.selected_text = sample.selected_text.map(post_process)
sample.to_csv("../input/submission101.csv", index=False)
