# import torch
# import numpy as np
# # print(torch.cuda.is_available())
# # print(torch.cuda.get_device_name(0))
# x = np.array(2,128)
# print(x)
# print(x.shape)
# start_logits, end_logits = x.split(1, dim=-1)
# print(start_logits)
# start_logits = start_logits.squeeze(-1)
# print("start_logits = ",start_logits,start_logits.shape )
# print(np.vstack(start_logits).shape)
# print()
# # end_logits = end_logits.squeeze(-1


# # p = np.array([[  101, 17111,  2080,  6517,  1045,  2097,  3335,  2017,  2182,  1999,
# #          2624,  5277,   999,   999,   999,   102,     0,     0,     0,     0,
# #             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
# #             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
# #             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
# #             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
# #             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
# #             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
# #             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
# #             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
# #             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
# #             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
# #             0,     0,     0,     0,     0,     0,     0,     0],[  101, 17111,  2080,  6517,  1045,  2097,  3335,  2017,  2182,  1999,
# #          2624,  5277,   999,   999,   999,   102,     0,     0,     0,     0,
# #             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
# #             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
# #             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
# #             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
# #             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
# #             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
# #             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
# #             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
# #             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
# #             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
# #             0,     0,     0,     0,     0,     0,     0,     0]])
# # kk = p.reshape(2,128)
# # mask_start = kk[0, 3:-1][:-112] 
# # print(mask_start)
# # print(np.nonzero(mask_start)[0])
# # import config
# # import tokenizers
# # tt = " Sooo SAD I will miss you here in San Diego!!!'?"
# # ss = "positive"
# # p = config.TOKENIZER.encode(ss,tt)
# # print(p.tokens)
# # print(p.ids)
# # print(p.offsets)
# # print(tokenizers.__version__) 
import numpy as np 
  
# p = [np.array([[1,2,3],
#      [4,5,6]])]
# print(p)
# kk = np.vstack(p)
# print(kk)
# print(kk.shape)

# import pandas as pd 




# # initialize list of lists 
# data = [["6ce4a4954b","Sooo I will SAD you here in San Diego!!!","Sooo SAD", "negative"],
#        ["6ce4a4954c","Sooo I SAD will you here in San Diego!!!","Sooo SAD", "negative"]] 
  
# # Create the pandas DataFrame 
# df = pd.DataFrame(data, columns = ["textID","text","selected_text","sentiment"]) 
  
# # print dataframe. 
# print(df)
from settings import get_module_logger


logger = get_module_logger(__name__)

logger.info("TEST HERE")